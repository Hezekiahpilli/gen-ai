"""
Reset Vector Database
Clears the existing vector database so it can be rebuilt with improved chunking
"""

import shutil
import os

def reset_db():
    db_path = "./chroma_db"
    
    if os.path.exists(db_path):
        print(f"Removing existing database at {db_path}...")
        shutil.rmtree(db_path)
        print("Database removed successfully!")
    else:
        print("No existing database found.")
    
    print("\nThe database will be rebuilt when you run the application next time.")
    print("This will ensure all documents are processed with the improved chunking strategy.")

if __name__ == "__main__":
    reset_db()

