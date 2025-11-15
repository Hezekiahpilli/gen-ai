# Sample Data Files

This directory contains sample data files for testing the Document Assistant system.

## Files Included

### CSV Files (Structured Data)

* **pharmaceuticals.csv** - Sample pharmaceutical orders data

  * Contains order information including product names, quantities, status, and managers
  * Used for testing structured data queries

* **supplychain.csv** - Sample supply chain data

  * Contains work order information for various products
  * Includes planned release dates and product categories

### Adding Your Own Documents

To test with your own documents, add files to this directory:

#### Supported Formats:

* **PDF files** (.pdf) - For unstructured text documents
* **Word documents** (.docx) - For formatted documents
* **CSV files** (.csv) - For tabular data
* **Excel files** (.xlsx) - For spreadsheet data

#### Example Documents You Can Add:

* Insurance policies (PDF)
* HR hiring criteria (PDF/DOCX)
* Technical documentation (PDF/DOCX)
* Order data (CSV/Excel)
* Any other business documents

## Sample Questions

Based on the sample data provided:

1. What is the order quantity handled by Ranjit?
2. What is the percentage of orders that haven't dispatched?
3. List of products under the recoil kits orders?
4. How much GST was charged for my insurance, and when does my third-party insurance expire?
5. What are the roles we are currently hiring for?
6. For the product Glock 17, what is the planned WO release date?
7. Ok, try Glock - 17
8. What is our criteria to hire a data scientist?
9. What are the benefits of log book, and how can I set it up?
10. How do I create a zone?



Note

The current sample data is minimal for demonstration purposes. For production use:

* Add your actual business documents
* Ensure sensitive data is properly secured
* Test with representative data volumes
