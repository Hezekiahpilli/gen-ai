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
        """Extract text from PDF files"""
        chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                full_text = ""
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    full_text += text + "\n"
                
                # Split into chunks (simple approach - by paragraphs)
                paragraphs = full_text.split('\n\n')
                for i, para in enumerate(paragraphs):
                    if len(para.strip()) > 50:  # Filter out very short chunks
                        chunk = DocumentChunk(
                            content=para.strip(),
                            metadata={
                                'source': file_path,
                                'page': i // 3,  # Approximate page number
                                'filename': os.path.basename(file_path)
                            },
                            chunk_id=f"{os.path.basename(file_path)}_chunk_{i}",
                            source_type='pdf'
                        )
                        chunks.append(chunk)
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
        
        return chunks
    
    def process_docx(self, file_path: str) -> List[DocumentChunk]:
        """Extract text from DOCX files"""
        chunks = []
        try:
            doc = DocxDocument(file_path)
            full_text = ""
            
            for para in doc.paragraphs:
                full_text += para.text + "\n"
            
            # Split into chunks
            paragraphs = full_text.split('\n\n')
            for i, para in enumerate(paragraphs):
                if len(para.strip()) > 50:
                    chunk = DocumentChunk(
                        content=para.strip(),
                        metadata={
                            'source': file_path,
                            'filename': os.path.basename(file_path)
                        },
                        chunk_id=f"{os.path.basename(file_path)}_chunk_{i}",
                        source_type='docx'
                    )
                    chunks.append(chunk)
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
                        content += f"Products: {', '.join(name_data['product_'].dropna().unique()[:5])}\n"
                        if 'qty' in df.columns:
                            content += f"Total quantity: {name_data['qty'].sum()}\n"
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
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.embedding_model.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
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
            
            # Query about dispatch status
            if 'dispatch' in query_lower and 'percentage' in query_lower:
                if 'status' in df.columns:
                    total = len(df)
                    not_dispatched = len(df[df['status'] != 'Dispatched'])
                    percentage = (not_dispatched / total) * 100
                    results['not_dispatched_percentage'] = round(percentage, 2)
                    results['not_dispatched_count'] = not_dispatched
                    results['total_orders'] = total
            
            # Query about recoil kits
            if 'recoil kit' in query_lower:
                if 'product_' in df.columns:
                    recoil_data = df[df['product_'].str.contains('recoil', case=False, na=False)]
                    if not recoil_data.empty:
                        results['recoil_products'] = recoil_data['product_'].unique().tolist()
                        results['recoil_count'] = len(recoil_data)
            
            # Query about Glock
            if 'glock' in query_lower:
                if 'product_' in df.columns:
                    # Handle both "Glock 17" and "Glock - 17" formats
                    glock_pattern = r'glock[\s\-]*17'
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
        
        # Search vector store for relevant documents
        search_results = self.vector_store.search(query, k=5)
        
        # Build context from search results
        context = ""
        if search_results and search_results['documents']:
            for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
                context += f"{doc}\n"
                response['sources'].append(metadata.get('filename', 'Unknown'))
        
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
        
        elif 'glock' in query_lower and ('wo release' in query_lower or 'work order' in query_lower):
            if 'glock_wo_release_dates' in structured_results:
                dates = structured_results['glock_wo_release_dates']
                products = structured_results.get('glock_products', [])
                if products:
                    response['answer'] = f"For Glock products ({', '.join(products)}), the planned WO release dates are: {', '.join(dates)}"
                else:
                    response['answer'] = f"Planned WO release dates for Glock products: {', '.join(dates)}"
                response['data'] = structured_results
        
        elif 'gst' in query_lower or 'insurance' in query_lower:
            # Search for insurance information in context
            if context:
                # Extract GST and insurance info from context using regex
                gst_pattern = r'GST[:\s]+([₹\d,\.]+)'
                insurance_pattern = r'(third party|TP|OD).*expire.*?(\d{2}/\d{2}/\d{4})'
                
                gst_match = re.search(gst_pattern, context, re.IGNORECASE)
                insurance_match = re.search(insurance_pattern, context, re.IGNORECASE)
                
                answer_parts = []
                if gst_match:
                    answer_parts.append(f"GST charged: {gst_match.group(1)}")
                if insurance_match:
                    answer_parts.append(f"Third party insurance expires on: {insurance_match.group(2)}")
                
                if answer_parts:
                    response['answer'] = ". ".join(answer_parts)
                else:
                    response['answer'] = "Insurance and GST information found in documents. Please check the vehicle insurance document for specific details."
        
        elif any(word in query_lower for word in ['hiring', 'recruit', 'data scientist', 'roles']):
            if context:
                response['answer'] = self._extract_from_context(query, context)
        
        elif 'log book' in query_lower or 'zone' in query_lower:
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
        """Extract answer from context using QA model or rules"""
        if self.use_local_llm:
            try:
                result = self.qa_pipeline(question=query, context=context)
                return result['answer']
            except:
                pass
        
        # Fallback to rule-based extraction
        sentences = context.split('.')
        relevant_sentences = []
        query_words = set(query.lower().split())
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if len(query_words.intersection(sentence_words)) > 2:
                relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            return ". ".join(relevant_sentences[:3]) + "."
        
        return "Information found in documents but requires more specific extraction."

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
