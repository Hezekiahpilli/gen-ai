"""
Reset Vector Database
Clears the existing vector database so it can be rebuilt with improved chunking
"""

import shutil
import os
import time
import sys

def reset_db():
    db_path = "./chroma_db"
    
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}...")
        
        # Try multiple times with delay for Windows file locks
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(db_path)
                print("Database removed successfully!")
                break
            except PermissionError:
                if attempt < max_attempts - 1:
                    print(f"Database is locked (attempt {attempt + 1}/{max_attempts}). Waiting...")
                    time.sleep(2)
                else:
                    print("\n❌ ERROR: Database is being used by another process.")
                    print("\nPlease close any running instances of:")
                    print("  - streamlit app")
                    print("  - document_assistant.py")
                    print("  - any Jupyter notebooks")
                    print("\nThen run this script again.")
                    sys.exit(1)
            except Exception as e:
                print(f"Error removing database: {e}")
                sys.exit(1)
    else:
        print("No existing database found.")
    
    print("\n✅ The database will be rebuilt when you run the application next time.")
    print("This will ensure all documents are processed with the improved chunking strategy.")

if __name__ == "__main__":
    reset_db()

