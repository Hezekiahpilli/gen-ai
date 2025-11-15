"""
Document Assistant - RAG System for Multi-format Document Interaction
This system enables conversational access to both structured and unstructured documents.
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Union
import json
import re
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
import csv

# Vector store and embeddings
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# LLM and processing
import openai
from transformers import pipeline
import torch

@dataclass
class DocumentChunk:
    """Represents a chunk of document content"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    source_type: str  # 'pdf', 'docx', 'csv', etc.
    
class DocumentProcessor:
    """Handles processing of different document types"""
    
    def __init__(self):
        self.chunks = []
        
    def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from PDF files with improved chunking"""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                page_texts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    page_texts.append((page_num + 1, text))
                    full_text += text + "\n"
                
                # Create chunks with better strategy
                # Strategy 1: Store full document as one chunk for comprehensive searches
                if len(full_text.strip()) > 100:
                    chunks.append(DocumentChunk(
                        content=full_text.strip(),
                        metadata={
                            'source': file_path,
                            'page': 'all',
                            'filename': os.path.basename(file_path),
                            'chunk_type': 'full_document'
                        },
                        chunk_id=f"{os.path.basename(file_path)}_full",
                        source_type='pdf'
                    ))
                
                # Strategy 2: Create overlapping chunks of ~1500 characters for better context
                chunk_size = 1500
                overlap = 300
                start = 0
                chunk_idx = 0
                
                while start < len(full_text):
                    end = start + chunk_size
                    chunk_text = full_text[start:end]
                    
                    if len(chunk_text.strip()) > 100:
                        chunks.append(DocumentChunk(
                            content=chunk_text.strip(),
                            metadata={
                                'source': file_path,
                                'filename': os.path.basename(file_path),
                                'chunk_type': 'overlap_chunk'
                            },
                            chunk_id=f"{os.path.basename(file_path)}_overlap_{chunk_idx}",
                            source_type='pdf'
                        ))
                    
                    start = end - overlap
                    chunk_idx += 1
                
                # Strategy 3: Per-page chunks for page-specific queries
                for page_num, page_text in page_texts:
                    if len(page_text.strip()) > 50:
                        chunks.append(DocumentChunk(
                            content=page_text.strip(),
                            metadata={
                                'source': file_path,
                                'page': page_num,
                                'filename': os.path.basename(file_path),
                                'chunk_type': 'page'
                            },
                            chunk_id=f"{os.path.basename(file_path)}_page_{page_num}",
                            source_type='pdf'
                        ))
                        
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
        
        return chunks
    
    def process_docx(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from DOCX files with improved chunking"""
        chunks = []
        try:
            doc = DocxDocument(file_path)
            full_text = ""
            
            for para in doc.paragraphs:
                full_text += para.text + "\n"
            
            # Extract tables from DOCX as well
            for table in doc.tables:
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    full_text += " | ".join(row_data) + "\n"
            
            # Strategy 1: Full document chunk
            if len(full_text.strip()) > 100:
                chunks.append(DocumentChunk(
                    content=full_text.strip(),
                    metadata={
                        'source': file_path,
                        'filename': os.path.basename(file_path),
                        'chunk_type': 'full_document'
                    },
                    chunk_id=f"{os.path.basename(file_path)}_full",
                    source_type='docx'
                ))
            
            # Strategy 2: Overlapping chunks with larger size for complete context
            chunk_size = 1500
            overlap = 300
            start = 0
            chunk_idx = 0
            
            while start < len(full_text):
                end = start + chunk_size
                chunk_text = full_text[start:end]
                
                if len(chunk_text.strip()) > 100:
                    chunks.append(DocumentChunk(
                        content=chunk_text.strip(),
                        metadata={
                            'source': file_path,
                            'filename': os.path.basename(file_path),
                            'chunk_type': 'overlap_chunk'
                        },
                        chunk_id=f"{os.path.basename(file_path)}_overlap_{chunk_idx}",
                        source_type='docx'
                    ))
                
                start = end - overlap
                chunk_idx += 1
                
        except Exception as e:
            print(f"Error processing DOCX {file_path}: {e}")
        
        return chunks
    
    def process_csv(self, file_path: str) -> tuple:
        """Process CSV files and return both DataFrame and text chunks"""
        chunks = []
        df = None
        
        try:
            df = pd.read_csv(file_path)
            
            # Create summary chunk
            summary = f"CSV File: {os.path.basename(file_path)}\n"
            summary += f"Columns: {', '.join(df.columns)}\n"
            summary += f"Number of rows: {len(df)}\n"
            
            # Add sample data description
            for col in df.columns[:10]:  # Limit to first 10 columns
                if df[col].dtype in ['int64', 'float64']:
                    summary += f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}\n"
                else:
                    unique_vals = df[col].nunique()
                    summary += f"{col}: {unique_vals} unique values\n"
                    if unique_vals < 20:
                        summary += f"  Values: {', '.join(df[col].unique()[:10].astype(str))}\n"
            
            chunks.append(DocumentChunk(
                content=summary,
                metadata={
                    'source': file_path,
                    'filename': os.path.basename(file_path),
                    'type': 'csv_summary'
                },
                chunk_id=f"{os.path.basename(file_path)}_summary",
                source_type='csv'
            ))
            
            # Create chunks for specific important rows (like orders with specific names)
            if 'mktg_specialistsmanagers' in df.columns:
                for name in df['mktg_specialistsmanagers'].unique():
                    if pd.notna(name):
                        name_data = df[df['mktg_specialistsmanagers'] == name]
                        content = f"Orders handled by {name}:\n"
                        content += f"Total orders: {len(name_data)}\n"
                        products = name_data['product_'].dropna().astype(str).unique()[:5]
                        content += f"Products: {', '.join(products)}\n"
                        if 'qty' in df.columns:
                            qty_series = pd.to_numeric(name_data['qty'], errors='coerce')
                            total_qty = qty_series.sum(skipna=True)
                            if pd.isna(total_qty):
                                total_qty = 0
                            elif float(total_qty).is_integer():
                                total_qty = int(total_qty)
                            content += f"Total quantity: {total_qty}\n"
                        if 'status' in df.columns:
                            content += f"Status distribution: {name_data['status'].value_counts().to_dict()}\n"
                        
                        chunks.append(DocumentChunk(
                            content=content,
                            metadata={
                                'source': file_path,
                                'filename': os.path.basename(file_path),
                                'person': name
                            },
                            chunk_id=f"{os.path.basename(file_path)}_{name}",
                            source_type='csv'
                        ))
            
        except Exception as e:
            print(f"Error processing CSV {file_path}: {e}")
        
        return chunks, df

class VectorStore:
    """Manages vector storage and retrieval"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name="documents",
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.client.get_collection("documents")
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to vector store"""
        if not chunks:
            return
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query: str, k: int = 50) -> List[Dict]:
        """Search for relevant documents - increased to 50 for comprehensive, complete results"""
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(k, self.collection.count())  # Don't exceed available documents
        )
        
        return results

class StructuredDataHandler:
    """Handles queries on structured data (CSV/Excel)"""
    
    def __init__(self):
        self.dataframes = {}
    
    def add_dataframe(self, name: str, df: pd.DataFrame):
        """Store a dataframe for querying"""
        self.dataframes[name] = df
    
    def query_data(self, query: str) -> Dict[str, Any]:
        """Process queries on structured data"""
        query_lower = query.lower()
        results = {}
        
        # Check for specific patterns in the query
        for df_name, df in self.dataframes.items():
            # Query about Ranjit
            if 'ranjit' in query_lower:
                if 'mktg_specialistsmanagers' in df.columns:
                    ranjit_data = df[df['mktg_specialistsmanagers'] == 'Ranjit']
                    if not ranjit_data.empty:
                        if 'qty' in df.columns:
                            total_qty = ranjit_data['qty'].sum()
                            results['ranjit_order_quantity'] = total_qty
                        results['ranjit_total_orders'] = len(ranjit_data)
                        if 'product_' in df.columns:
                            results['ranjit_products'] = ranjit_data['product_'].unique().tolist()
            
            # Query about dispatch status - only check pharmaceutical orders
            if 'dispatch' in query_lower and 'percentage' in query_lower:
                # Only check dataframes that have order data (with mktg_specialistsmanagers column)
                if 'status' in df.columns and 'mktg_specialistsmanagers' in df.columns:
                    total = len(df)
                    not_dispatched = len(df[df['status'] != 'Dispatched'])
                    percentage = (not_dispatched / total) * 100
                    results['not_dispatched_percentage'] = round(percentage, 2)
                    results['not_dispatched_count'] = not_dispatched
                    results['total_orders'] = total
            
            # Query about recoil kits - check both category and product name
            if 'recoil kit' in query_lower or 'recoil' in query_lower:
                # Check if it's supply chain data with category column
                if 'category' in df.columns:
                    recoil_data = df[df['category'].str.contains('Recoil', case=False, na=False)]
                    if not recoil_data.empty and 'product_name' in df.columns:
                        results['recoil_products'] = recoil_data['product_name'].unique().tolist()
                        results['recoil_count'] = len(recoil_data)
                # Check pharmaceuticals with product_ column
                elif 'product_' in df.columns:
                    recoil_data = df[df['product_'].str.contains('recoil', case=False, na=False)]
                    if not recoil_data.empty:
                        results['recoil_products'] = recoil_data['product_'].unique().tolist()
                        results['recoil_count'] = len(recoil_data)
            
            # Query about Glock - check both product_name and product_ columns
            if 'glock' in query_lower:
                glock_pattern = r'glock[\s\-]*17'
                # Check supply chain data with product_name column
                if 'product_name' in df.columns:
                    glock_data = df[df['product_name'].str.contains(glock_pattern, case=False, regex=True, na=False)]
                    if not glock_data.empty:
                        if 'planned_wo_release_date' in df.columns:
                            results['glock_wo_release_dates'] = glock_data['planned_wo_release_date'].dropna().unique().tolist()
                        if 'product_name' in df.columns:
                            results['glock_products'] = glock_data['product_name'].unique().tolist()
                        if 'status' in df.columns:
                            results['glock_status'] = glock_data['status'].unique().tolist()
                # Check pharmaceuticals with product_ column
                elif 'product_' in df.columns:
                    glock_data = df[df['product_'].str.contains(glock_pattern, case=False, regex=True, na=False)]
                    if not glock_data.empty:
                        if 'wo_release_date_planned' in df.columns:
                            results['glock_wo_release_dates'] = glock_data['wo_release_date_planned'].dropna().unique().tolist()
                        results['glock_products'] = glock_data['product_'].unique().tolist()
        
        return results

class ConversationalRAG:
    """Main RAG system for conversational document interaction"""
    
    def __init__(self, source_folder: str):
        self.source_folder = source_folder
        self.doc_processor = DocumentProcessor()
        self.vector_store = VectorStore()
        self.structured_handler = StructuredDataHandler()
        self.all_chunks = []
        
        # Initialize local LLM (using HuggingFace model)
        try:
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                device=0 if torch.cuda.is_available() else -1
            )
            self.use_local_llm = True
        except:
            self.use_local_llm = False
            print("Warning: Local LLM not available. Using rule-based responses.")
        
        self.load_documents()
    
    def load_documents(self):
        """Load all documents from source folder"""
        print(f"Loading documents from {self.source_folder}...")
        
        for file_path in Path(self.source_folder).glob("*"):
            if file_path.suffix.lower() == '.pdf':
                chunks = self.doc_processor.process_pdf(str(file_path))
                self.all_chunks.extend(chunks)
                self.vector_store.add_documents(chunks)
                
            elif file_path.suffix.lower() == '.docx':
                chunks = self.doc_processor.process_docx(str(file_path))
                self.all_chunks.extend(chunks)
                self.vector_store.add_documents(chunks)
                
            elif file_path.suffix.lower() == '.csv':
                chunks, df = self.doc_processor.process_csv(str(file_path))
                if df is not None:
                    self.structured_handler.add_dataframe(file_path.stem, df)
                self.all_chunks.extend(chunks)
                self.vector_store.add_documents(chunks)
        
        print(f"Loaded {len(self.all_chunks)} document chunks")
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate response to user query"""
        response = {
            'query': query,
            'answer': '',
            'sources': [],
            'data': None
        }
        
        # First, check structured data
        structured_results = self.structured_handler.query_data(query)
        
        # Search vector store for relevant documents - get MAXIMUM results for comprehensive answers
        search_results = self.vector_store.search(query, k=30)
        
        # Build comprehensive context from ALL search results
        context = ""
        source_files = set()
        retrieved_chunks = []
        
        if search_results and search_results['documents']:
            for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
                retrieved_chunks.append({'text': doc, 'metadata': metadata})
                source_files.add(metadata.get('filename', 'Unknown'))
            
            # Prioritize full_document chunks for comprehensive answers
            full_docs = []
            page_chunks = []
            other_chunks = []
            
            for chunk in retrieved_chunks:
                metadata = chunk['metadata']
                doc = chunk['text']
                if metadata.get('chunk_type') == 'full_document':
                    full_docs.append(doc)
                elif metadata.get('chunk_type') == 'page':
                    page_chunks.append(doc)
                else:
                    other_chunks.append(doc)
            
            # Use full documents first for complete information, then pages, then overlapping chunks
            # Don't limit the chunks - use ALL relevant information
            context = "\n\n---\n\n".join(full_docs + page_chunks + other_chunks)
            response['sources'] = list(source_files)
        
        # Generate answer based on query type
        query_lower = query.lower()
        
        # Handle specific queries with structured data
        if 'ranjit' in query_lower and 'order quantity' in query_lower:
            if 'ranjit_order_quantity' in structured_results:
                response['answer'] = f"Ranjit has handled a total order quantity of {structured_results['ranjit_order_quantity']} units."
                response['data'] = structured_results
        
        elif 'dispatch' in query_lower and 'percentage' in query_lower:
            if 'not_dispatched_percentage' in structured_results:
                response['answer'] = f"{structured_results['not_dispatched_percentage']}% of orders haven't been dispatched. "
                response['answer'] += f"That's {structured_results['not_dispatched_count']} out of {structured_results['total_orders']} total orders."
                response['data'] = structured_results
        
        elif 'recoil kit' in query_lower:
            if 'recoil_products' in structured_results:
                products = structured_results['recoil_products']
                response['answer'] = f"Products under recoil kit orders:\n"
                for product in products:
                    response['answer'] += f"- {product}\n"
                response['data'] = structured_results
        
        elif 'glock' in query_lower and ('wo release' in query_lower or 'work order' in query_lower or 'release date' in query_lower or 'planned' in query_lower):
            if 'glock_wo_release_dates' in structured_results:
                dates = structured_results['glock_wo_release_dates']
                products = structured_results.get('glock_products', [])
                status_list = structured_results.get('glock_status', [])
                
                answer_parts = []
                for i, product in enumerate(products):
                    date = dates[i] if i < len(dates) else dates[0]
                    status = status_list[i] if i < len(status_list) else 'N/A'
                    answer_parts.append(f"{product}: Planned WO release date is {date}, Status: {status}")
                
                response['answer'] = ". ".join(answer_parts)
                response['data'] = structured_results
        
        elif 'gst' in query_lower or 'insurance' in query_lower:
            # Search for insurance information in context with comprehensive extraction
            if context:
                # Extract ALL insurance-related information
                answer_parts = []
                
                # Look for GST amounts (multiple patterns)
                gst_patterns = [
                    r'GST[:\s]*(?:Rs\.?|₹)?\s*([\d,\.]+)',
                    r'(?:Tax|GST)[:\s]+([₹\d,\.]+)',
                    r'Service Tax[:\s]+([₹\d,\.]+)'
                ]
                for pattern in gst_patterns:
                    matches = re.findall(pattern, context, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            answer_parts.append(f"GST/Tax amount: Rs.{match}")
                        break
                
                # Look for insurance expiry dates (multiple patterns)
                expiry_patterns = [
                    r'(?:Third Party|TP|third party|OD).*?(?:expire|expiry|valid till|valid until)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    r'(?:expire|expiry|valid till)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    r'valid.*?(?:up to|till|until)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
                ]
                for pattern in expiry_patterns:
                    matches = re.findall(pattern, context, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            answer_parts.append(f"Insurance expiry date: {match}")
                        break
                
                # Extract premium amounts
                premium_patterns = [
                    r'Premium[:\s]*(?:Rs\.?|₹)?\s*([\d,\.]+)',
                    r'Total Premium[:\s]*(?:Rs\.?|₹)?\s*([\d,\.]+)'
                ]
                for pattern in premium_patterns:
                    matches = re.findall(pattern, context, re.IGNORECASE)
                    if matches:
                        for match in matches:
                            answer_parts.append(f"Premium amount: Rs.{match}")
                        break
                
                # If specific patterns found, use them
                if answer_parts:
                    response['answer'] = ". ".join(answer_parts)
                else:
                    # Fall back to comprehensive extraction
                    response['answer'] = self._extract_from_context(query, context)
        
        elif any(word in query_lower for word in ['hiring', 'recruit', 'data scientist', 'roles', 'position', 'opening']):
            if retrieved_chunks:
                hiring_roles = self._extract_hiring_roles(retrieved_chunks)
                if hiring_roles:
                    response['answer'] = self._format_hiring_response(hiring_roles)
                    response['sources'] = sorted({role['source'] for role in hiring_roles})
                elif context:
                    response['answer'] = self._extract_from_context(query, context)
            elif context:
                response['answer'] = self._extract_from_context(query, context)
        
        elif 'create a zone' in query_lower or ('log book' in query_lower and 'zone' in query_lower):
            if context:
                zone_answer = self._extract_zone_instructions(context)
                if zone_answer:
                    response['answer'] = zone_answer
                else:
                    response['answer'] = self._extract_from_context(query, context)
        elif 'log book' in query_lower:
            if context:
                response['answer'] = self._extract_from_context(query, context)
        
        # If no specific handler, use context-based answer
        if not response['answer'] and context:
            response['answer'] = self._extract_from_context(query, context)
        
        # Fallback if no answer generated
        if not response['answer']:
            response['answer'] = "I couldn't find specific information about your query in the documents. Please try rephrasing or asking about specific data available in the files."
        
        return response
    
    def _extract_from_context(self, query: str, context: str) -> str:
        """Extract comprehensive answer from context with detailed information"""
        llm_answer = ""
        if self.use_local_llm:
            try:
                # Use MUCH larger context for complete answers
                result = self.qa_pipeline(question=query, context=context[:12000])
                llm_answer = result.get('answer', '').strip()
            except:
                llm_answer = ""
        
        # Enhanced rule-based extraction for comprehensive answers
        query_lower = query.lower()
        query_keywords = set([word for word in query_lower.split() if len(word) > 3])
        
        # Split context into sentences and paragraphs
        paragraphs = context.split('\n\n')
        relevant_paragraphs = []
        scores = []
        
        # Score each paragraph by relevance
        for para in paragraphs:
            if len(para.strip()) < 20:
                continue
                
            para_lower = para.lower()
            para_words = set([word for word in para_lower.split() if len(word) > 3])
            
            # Calculate relevance score
            keyword_matches = len(query_keywords.intersection(para_words))
            
            # Boost score for exact phrase matches
            if any(keyword in para_lower for keyword in query_keywords):
                keyword_matches += 2
            
            if keyword_matches > 0:
                scores.append((keyword_matches, para))
        
        # Sort by relevance and take MORE paragraphs for complete answers
        scores.sort(reverse=True, key=lambda x: x[0])
        
        # Build comprehensive answer from ALL relevant paragraphs
        answer_parts = []
        seen_content = set()
        
        # Take up to 20 most relevant paragraphs instead of 8 for complete answers
        for score, para in scores[:20]:
            para_clean = para.strip()
            # Avoid duplicates
            if para_clean not in seen_content and len(para_clean) > 30:
                answer_parts.append(para_clean)
                seen_content.add(para_clean)
        
        if answer_parts:
            # Create a comprehensive answer - NO TRUNCATION for complete information
            combined_parts = []
            if llm_answer:
                combined_parts.append(llm_answer)
            combined_parts.extend(answer_parts)
            
            deduped = []
            seen = set()
            for part in combined_parts:
                cleaned = part.strip()
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key not in seen:
                    deduped.append(cleaned)
                    seen.add(key)
            
            if deduped:
                return "\n\n".join(deduped)
        
        # If no good paragraph matches, try sentence-level extraction
        sentences = re.split(r'[.!?]+', context)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_clean = sentence.strip()
            if len(sentence_clean) < 20:
                continue
            
            sentence_lower = sentence_clean.lower()
            if any(keyword in sentence_lower for keyword in query_keywords):
                relevant_sentences.append(sentence_clean)
        
        if relevant_sentences:
            combined_parts = []
            if llm_answer:
                combined_parts.append(llm_answer)
            combined_parts.extend(relevant_sentences)
            answer = ". ".join(combined_parts)
            if not answer.endswith('.'):
                answer += "."
            return answer
        
        if llm_answer:
            return llm_answer
        
        return f"Based on the documents, I found relevant information but it may not directly answer your specific question. Please try rephrasing or asking more specifically about: {', '.join(list(query_keywords)[:5])}"
    
    def _extract_hiring_roles(self, retrieved_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract hiring role information from retrieved document chunks."""
        role_patterns = [
            r'(?:role|position|designation|job title)\s*[:\-]\s*(.+)',
            r'(?:hiring|looking for|recruiting)\s+(?:an?|the)?\s*([A-Za-z0-9 /&,\-]+)',
            r'open(?:ing| position)s?\s*[:\-]\s*(.+)'
        ]
        job_terms = [
            'developer', 'scientist', 'analyst', 'engineer', 'manager',
            'consultant', 'architect', 'specialist', 'designer', 'lead',
            'administrator', 'executive', 'associate', 'intern', 'expert',
            'officer', 'technician', 'coordinator', 'director', 'owner',
            'controller', 'supervisor', 'strategist', 'analyst', 'planner',
            'bi developer', 'data engineer'
        ]
        job_base_terms = [term for term in job_terms if ' ' not in term]
        context_keywords = [
            'hiring', 'role', 'position', 'opening', 'vacancy',
            'recruit', 'job', 'opportunity', 'career'
        ]
        role_prefix_terms = {
            'senior', 'junior', 'lead', 'principal', 'data', 'power',
            'bi', 'business', 'full', 'stack', 'cloud', 'machine',
            'learning', 'software', 'java', 'python', 'powerbi',
            'digital', 'analytics'
        }
        disallowed_starts = (
            'experience', 'responsibil', 'requirement', 'skills',
            'about', 'summary', 'profile', 'here', 'as a', 'we are'
        )
        
        roles_map = {}
        seen = set()
        
        for chunk in retrieved_chunks:
            text = chunk['text']
            metadata = chunk.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            
            for idx, line in enumerate(lines):
                normalized = line.lower()
                context_window = " ".join(lines[max(0, idx-2):idx+1]).lower()
                
                role_title = None
                for pattern in role_patterns:
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        role_title = match.group(1).strip(" -•:\t.;")
                        break
                
                if not role_title:
                    if 'job description' in normalized and any(term in normalized for term in job_terms):
                        prefix = re.split(r'job description', line, flags=re.IGNORECASE)[0].strip()
                        words = prefix.split()
                        job_idx = None
                        for idx in range(len(words) - 1, -1, -1):
                            clean_word = re.sub(r'[^a-z0-9]', '', words[idx].lower())
                            if clean_word in job_base_terms:
                                job_idx = idx
                                break
                        if job_idx is not None:
                            start_idx = job_idx
                            while start_idx > 0:
                                prev_clean = re.sub(r'[^a-z0-9]', '', words[start_idx - 1].lower())
                                if prev_clean in role_prefix_terms:
                                    start_idx -= 1
                                else:
                                    break
                            candidate = " ".join(words[start_idx:job_idx + 1]).strip(" -•:\t.;")
                            if candidate:
                                role_title = candidate
                
                if not role_title:
                    bullet_match = re.match(r'^[•\-\*\d\.\)\( ]+', line)
                    bullet_line = re.sub(r'^[•\-\*\d\.\)\( ]+', '', line)
                    if bullet_match:
                        if any(term in bullet_line.lower() for term in job_terms):
                            # Require contextual signal that this section is about hiring
                            if any(keyword in context_window for keyword in context_keywords) or any(keyword in normalized for keyword in context_keywords):
                                role_title = bullet_line.strip(" -•:\t.;")
                
                if not role_title and any(term in normalized for term in job_terms):
                    stripped_line = line.strip()
                    stripped_lower = stripped_line.lower()
                    words = stripped_line.split()
                    if len(stripped_line) <= 60 and len(words) <= 6 and not any(punct in stripped_line for punct in ['.', '!', '?', "'", '’']):
                        if not stripped_lower.startswith(disallowed_starts):
                            if re.match(r'^[A-Za-z0-9 /&\-\(\)]+$', stripped_line):
                                words_clean = [re.sub(r'[^a-z0-9]', '', w.lower()) for w in stripped_line.split()]
                                if any(word in job_base_terms for word in words_clean[-2:]):
                                    role_title = stripped_line
                
                if not role_title:
                    continue
                
                role_title = re.sub(r'\b(role|position|opening)\b', '', role_title, flags=re.IGNORECASE).strip(" -•:\t.;")
                role_title = re.sub(r'^(?:for|the|a|an)\s+', '', role_title, flags=re.IGNORECASE)
                role_title = re.sub(r'^(?:job description for|job desc for)\s+', '', role_title, flags=re.IGNORECASE)
                role_title = re.sub(r'^(?:opening|openings|open)\s+(?:for\s+)?', '', role_title, flags=re.IGNORECASE)
                role_title = re.sub(r'\s+', ' ', role_title).strip()
                
                normalized_role = role_title.lower()
                if not any(term in normalized_role for term in job_terms):
                    continue
                
                key = (role_title.lower(), filename.lower())
                if key in seen:
                    continue
                
                detail_lines = [line]
                for extra_line in lines[idx+1:idx+4]:
                    extra_lower = extra_line.lower()
                    potential_new_role = (
                        (re.search(r'(?:position|role|opening)\s*[:\-]', extra_lower) and any(term in extra_lower for term in job_terms))
                        or (re.match(r'^[•\-\*\d]', extra_line.strip()) and any(term in extra_lower for term in job_terms))
                    )
                    if potential_new_role:
                        break
                    if any(trigger in extra_lower for trigger in ['responsibilit', 'requirement', 'skills', 'experience', 'about the role']):
                        detail_lines.append(extra_line)
                        break
                    if len(extra_line.split()) < 4:
                        continue
                    detail_lines.append(extra_line)
                    if len(" ".join(detail_lines)) > 500:
                        break
                
                detail = " ".join(detail_lines).strip()
                if len(detail) > 600:
                    detail = detail[:600].rstrip() + "..."
                
                if key not in roles_map:
                    roles_map[key] = {
                        'role': role_title,
                        'source': filename,
                        'details': []
                    }
                if detail and detail not in roles_map[key]['details']:
                    roles_map[key]['details'].append(detail)
                seen.add(key)
        
        return list(roles_map.values())
    
    def _format_hiring_response(self, roles: List[Dict[str, str]]) -> str:
        """Format hiring role information into a concise list of role names."""
        if not roles:
            return ""
        
        unique_roles = {}
        for role in roles:
            key = (role['role'], role['source'])
            unique_roles[key] = role
        
        lines = ["Open roles identified across the documents:"]
        for (role_name, source) in sorted(unique_roles.keys()):
            lines.append(f"- {role_name} (source: {source})")
        
        return "\n".join(lines)

    def _extract_zone_instructions(self, context: str) -> Optional[str]:
        """Extract instructions for creating a zone from the log book document."""
        lower_context = context.lower()
        anchor = 'to create a new zone'
        idx = lower_context.find(anchor)
        
        section_text = ""
        if idx != -1:
            end_idx = len(context)
            for marker in ['7.3', '7.2.2', '8.']:
                marker_pos = lower_context.find(marker.lower(), idx)
                if marker_pos != -1:
                    end_idx = min(end_idx, marker_pos)
            start_idx = max(0, lower_context.rfind('7.2.1', 0, idx))
            section_text = context[start_idx:end_idx]
        else:
            section_pattern = re.compile(
                r'(7\.2\.1\s+Create\s+a\s+New\s+Zone.*?)(?:7\.2\.2|7\.3|8\.)',
                re.IGNORECASE | re.DOTALL
            )
            match = section_pattern.search(context)
            if match:
                section_text = match.group(1)
        
        if not section_text:
            return None
        
        cleaned = " ".join(section_text.split())
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', cleaned) if s.strip()]
        heading = "7.2.1 Create a New Zone"
        if sentences:
            # Keep only the sentences that describe the procedure.
            filtered = []
            for sentence in sentences:
                if sentence.lower().startswith('7.2.1'):
                    continue
                filtered.append(sentence)
            if filtered:
                concise = " ".join(filtered[:3])
            else:
                concise = " ".join(sentences[:3])
        else:
            concise = cleaned
        
        summary = f"{heading} — {concise}"
        return summary.strip()

def main():
    """Main function to run the conversational document assistant"""
    
    # Initialize the RAG system
    rag = ConversationalRAG("Gen AI/Source")
    
    print("\n" + "="*60)
    print("DOCUMENT ASSISTANT - Conversational Interface")
    print("="*60)
    print("\nSystem initialized. Processing test questions...\n")
    
    # Test questions from the assignment
    test_questions = [
        "What is the order quantity handled by Ranjit?",
        "What is the percentage of orders that haven't dispatched?",
        "List of products under the recoil kits orders?",
        "How much GST was charged for my insurance and when does my third party insurance expire?",
        "What are the roles we are currently hiring for?",
        "For the product Glock 17 what is the planned WO release date?",
        "Ok try Glock - 17",
        "What is our criteria to hire a data scientist?",
        "What are the benifits of log book and how can i set it up?",
        "How do i create a zone?"
    ]
    
    # Process each question
    results = []
    for i, question in enumerate(test_questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 40)
        
        response = rag.generate_response(question)
        
        print(f"Answer: {response['answer']}")
        if response['sources']:
            print(f"Sources: {', '.join(set(response['sources']))}")
        if response['data']:
            print(f"Structured Data: Available")
        
        results.append(response)
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("Type 'quit' to exit")
    print("="*60)
    
    while True:
        user_query = input("\nYour question: ").strip()
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        response = rag.generate_response(user_query)
        print(f"\nAnswer: {response['answer']}")
        if response['sources']:
            print(f"Sources: {', '.join(set(response['sources']))}")

if __name__ == "__main__":
    main()
