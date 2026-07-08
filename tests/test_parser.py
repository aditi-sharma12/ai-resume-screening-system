import pytest
from src.parser import extract_text_from_pdf

def test_extract_text_from_pdf_invalid_path():
    """Verify that attempting to parse a non-existent file raises a FileNotFoundError."""
    with pytest.raises(Exception):
        extract_text_from_pdf("non_existent_file_path_12345.pdf")
