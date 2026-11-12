#to parse both tabular data
#(like CTCAE tables) and regular text blocks from the source PDFs.


import fitz  # PyMuPDF
import pdfplumber
import json

def extract_text_by_sections(pdf_path):
    """Extracts text while maintaining block reading order."""
    doc = fitz.open(pdf_path)
    extracted_data = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("blocks")  # Returns tuples of (x0, y0, x1, y1, text, block_no, block_type)
        for b in blocks:
            text = b[4].strip()
            if text:
                extracted_data.append({
                    "page": page_num + 1,
                    "text": text
                })
    return extracted_data

def extract_tables(pdf_path):
    """Extracts structured tables from the PDF."""
    tables_data = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                tables_data.append({
                    "page": page_idx + 1,
                    "table": table
                })
    return tables_data

if __name__ == "__main__":
    pdf_file = "data/guidelines/ASCO_2017_Cardiac_Dysfunction.pdf"
    
    parsed_text = extract_text_by_sections(pdf_file)
    with open("data/parsed_raw_text.json", "w") as f:
        json.dump(parsed_text, f, indent=2)
        
    print("PDF parsing complete. Output saved to data/parsed_raw_text.json")