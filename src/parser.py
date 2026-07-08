from pypdf import PdfReader

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts plain text content from a PDF file.
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text
