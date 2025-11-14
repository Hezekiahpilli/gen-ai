"""
Test script to verify improved RAG system with comprehensive answers
"""
import sys
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from document_assistant import ConversationalRAG
import time

print("="*80)
print("TESTING IMPROVED DOCUMENT ASSISTANT")
print("="*80)

# Initialize RAG system - this will load all documents with improved chunking
print("\nInitializing RAG system and loading documents...")
start_time = time.time()
rag = ConversationalRAG("Gen AI/Source")
load_time = time.time() - start_time

print(f"[OK] Documents loaded in {load_time:.2f} seconds")
print(f"[OK] Total chunks created: {len(rag.all_chunks)}")

# Test comprehensive answers
test_questions = [
    "How much GST was charged for my insurance and when does my third party insurance expire?",
    "What are the roles we are currently hiring for?",
    "What is our criteria to hire a data scientist?",
    "What are the benefits of log book and how can i set it up?",
    "How do i create a zone?"
]

print("\n" + "="*80)
print("TESTING COMPREHENSIVE ANSWERS FROM ACTUAL DOCUMENTS")
print("="*80)

for i, question in enumerate(test_questions, 1):
    print(f"\n{'='*80}")
    print(f"Question {i}: {question}")
    print('='*80)
    
    response = rag.generate_response(question)
    
    print(f"\nANSWER:")
    print(response['answer'])
    
    print(f"\nSOURCES: {', '.join(set(response['sources']))}")
    print(f"Answer length: {len(response['answer'])} characters")
    
    print("\n" + "-"*80)

print("\n" + "="*80)
print("TEST COMPLETED")
print("="*80)
print("\nThe system should now provide detailed, comprehensive answers")
print("from the actual PDF and DOCX documents.")

