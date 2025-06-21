import random
from typing import List
from pypdf import PdfReader

from evaluation import evaluate

def predict(pdf_path: str) -> List[int]:
    """
    Given the file path to a PDF, count its pages and return a random
    0/1 prediction for each page.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A list of integers (0 or 1), length == number of pages in the PDF.
    """
    # Load the PDF with pypdf
    reader = PdfReader(pdf_path, strict=False)
    num_pages = len(reader.pages)

    # Randomly choose 0 or 1 for each page
    return [random.randint(0, 1) for _ in range(num_pages)]

if __name__ == "__main__":
    evaluate(predict, split="test", n=10)
