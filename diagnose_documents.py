"""
Diagnostic script to see what's in the documents
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from document_assistant import DocumentProcessor
from pathlib import Path

processor = DocumentProcessor()

print("="*80)
print("DOCUMENT CONTENT DIAGNOSTIC")
print("="*80)

# Process each document
for file_path in Path("Gen AI/Source").glob("*"):
    if file_path.suffix.lower() in ['.pdf', '.docx']:
        print(f"\n{'='*80}")
        print(f"FILE: {file_path.name}")
        print('='*80)
        
        if file_path.suffix.lower() == '.pdf':
            chunks = processor.process_pdf(str(file_path))
        else:
            chunks = processor.process_docx(str(file_path))
        
        print(f"\nTotal chunks created: {len(chunks)}")
        
        # Show the full document chunk
        for chunk in chunks:
            if chunk.metadata.get('chunk_type') == 'full_document':
                print(f"\nFULL DOCUMENT CONTENT (first 2000 chars):")
                print("-"*80)
                content = chunk.content[:2000]
                print(content)
                print(f"\n... [Total length: {len(chunk.content)} characters]")
                break

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)

