# Document Assistant - Complete Usage Guide

## 🚀 Quick Start

### 1. Setup Your Documents

Place your documents in the `Gen AI/Source/` folder:

```
Gen AI/Source/
├── pharmaceuticals.csv         # Sample CSV data
├── supplychain.csv             # Sample supply chain data  
├── Doc 1.pdf                   # Your PDF documents
├── Doc 2.docx                  # Your DOCX documents
├── Doc 3.pdf                   # Insurance policies
├── Doc 4.pdf                   # HR documents
└── Doc 5.pdf                   # Technical documentation
```

### 2. Run the Application

**Option A: Demo Script (Quick Test)**
```bash
python demo_results.py
```

**Option B: Command Line Interface**
```bash
python document_assistant.py
```

**Option C: Web Interface (Recommended)**
```bash
streamlit run streamlit_app.py
```
Then open http://localhost:8501

## 📊 System Capabilities

### Complete Answer Generation

The system has been optimized to provide **complete, detailed answers** by:

1. **Larger Context Windows**: Uses up to 12,000 characters of context (3x more than before)
2. **More Chunks Retrieved**: Retrieves up to 50 relevant document chunks instead of 20
3. **Bigger Chunk Sizes**: Processes 1,500-character chunks instead of 800 (87.5% larger)
4. **No Truncation**: Returns complete answers without cutting off information
5. **Full Document Priority**: Prioritizes full document content for comprehensive responses

### Document Processing Strategy

The system uses a **three-tier chunking strategy** for maximum information retrieval:

1. **Full Document Chunks** - Complete document text for comprehensive answers
2. **Page-Level Chunks** - Individual pages for precise source attribution  
3. **Overlapping Chunks** - 1,500-char chunks with 300-char overlap for context continuity

## 💡 Best Practices

### Getting Complete Answers

1. **Be Specific**: Ask clear, specific questions
   - Good: "What is the GST amount charged for my insurance?"
   - Better: "What is the GST amount and when does my third party insurance expire?"

2. **Ask Follow-up Questions**: If the answer seems incomplete, ask for more details
   - "Can you provide more details about X?"
   - "What else should I know about Y?"

3. **Reference Documents**: Mention specific documents if you know them
   - "In Doc 1.pdf, what is the insurance expiry date?"

### Query Types Supported

#### 1. Structured Data Queries (CSV/Excel)
```
- "What is the total order quantity for Ranjit?"
- "What percentage of orders haven't been dispatched?"
- "List all products in the Recoil Kits category"
- "When is the planned WO release date for Glock 17?"
```

#### 2. Document Extraction Queries (PDF/DOCX)
```
- "What is the GST amount in my insurance document?"
- "When does my third party insurance expire?"
- "What are the hiring criteria for a data scientist?"
- "How do I set up a log book?"
```

#### 3. Combined/Hybrid Queries
```
- "Show me all orders and their insurance details"
- "Compare product quantities with their documentation"
```

## 🔧 Advanced Features

### Resetting the Database

If you add new documents or want to rebuild the index:

```bash
python reset_database.py
```

This will:
- Clear the existing vector database
- Force a complete rebuild on next run
- Apply any improvements to document processing

### Custom Document Addition

1. Add your files to `Gen AI/Source/`
2. Run `python reset_database.py`
3. Run the application - it will automatically process new files

### Supported File Types

| Format | Extension | Use Case |
|--------|-----------|----------|
| PDF | `.pdf` | Insurance docs, policies, reports |
| Word | `.docx` | HR documents, procedures, memos |
| CSV | `.csv` | Tabular data, orders, inventory |
| Excel | `.xlsx` | Spreadsheets, financial data |

## 📈 Performance Optimization

The improved system provides:

- **3x more context** for answer generation (12,000 vs 4,000 characters)
- **87% larger chunks** for better coherence (1,500 vs 800 characters)
- **2.5x more retrieval** (50 vs 20 chunks)
- **No answer truncation** - complete information every time
- **Better overlap** (300 vs 200 chars) for continuous context

## 🐛 Troubleshooting

### Partial Answers

If you're still getting partial answers:

1. **Reset the database**: `python reset_database.py`
2. **Check document quality**: Ensure PDFs are text-based (not scanned images)
3. **Ask more specifically**: Narrow down your question
4. **Check sources**: Verify the information exists in your documents

### Performance Issues

If the system is slow:

1. **Reduce documents**: Start with fewer documents to test
2. **Check file sizes**: Very large PDFs may take longer
3. **Use structured queries**: CSV/Excel queries are faster

### No Results Found

If the system can't find information:

1. **Verify files are in** `Gen AI/Source/` folder
2. **Check file format**: Ensure files are readable
3. **Run reset**: `python reset_database.py` to rebuild index
4. **Rephrase query**: Try different keywords

## 📝 Example Questions

### For Sample Data

Based on the included sample files:

**CSV Data Questions:**
```
1. "What is the order quantity handled by Ranjit?"
2. "What percentage of orders haven't been dispatched?"
3. "List all products under recoil kits"
4. "What is the planned release date for Glock 17?"
5. "Show me all orders with status Pending"
```

**PDF/DOCX Questions:**
```
1. "What is the GST charged for insurance?"
2. "When does the third party insurance expire?"
3. "What are the hiring criteria for data scientist?"
4. "How do I set up a log book?"
5. "What are the steps to create a zone?"
```

## 🎯 Tips for Best Results

1. **Reset After Adding Documents**: Always run `reset_database.py` after adding new documents
2. **Use Clear Language**: Simple, direct questions work best
3. **Be Patient**: First run takes longer as it builds the index
4. **Check Sources**: The system shows which files it used for answers
5. **Iterate**: If answer is incomplete, ask follow-up questions

## 🔐 Privacy & Security

- All processing happens **locally** on your machine
- No data is sent to external APIs (unless you configure OpenAI)
- Documents are only stored in your local vector database
- Database is in `./chroma_db/` folder

## 📚 Additional Resources

- **README.md** - Project overview and setup
- **ARCHITECTURE.md** - Technical architecture details
- **SOLUTION_SUMMARY.md** - Implementation approach
- **requirements.txt** - Python dependencies

## 🆘 Getting Help

For issues or questions:
1. Check this guide first
2. Review the troubleshooting section
3. Check the GitHub issues: https://github.com/Hezekiahpilli/gen-ai/issues
4. Create a new issue with details about your problem

---

**Remember**: The system now provides **complete, detailed answers** by using larger contexts, more chunks, and no truncation. If you're still getting partial answers, try resetting the database and ensuring your documents are properly formatted.

