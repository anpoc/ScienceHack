from typing import List
import fitz


def predict(pdf_path: str) -> List[int]:
    pdf = fitz.open(pdf_path)
    result = []
    for page in pdf:
        text = page.get_text()
        words = text.split()
        if len(words) > 800:
            result.append(0)
        else:
            result.append(1)
    return result

