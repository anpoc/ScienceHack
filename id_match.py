from typing import List
import fitz
import re


def pdf_to_pages(path: str) -> str:
    pdf = fitz.open(path)
    pages = []
    for page in pdf:
        pages.append(page.get_text())
    return pages


def get_numbers(text: str) -> List[int]:
    # match at least 5 characters composed of digits, A–Z, hyphen, slash or space
    pattern = r'(?:[0-9A-Z]|[\-/]){5,}'
    raw_matches = re.findall(pattern, text)

    blacklist = {"85630", "81827"}
    results: List[str] = []

    for m in raw_matches:
        # remove spaces, hyphens and slashes
        cleaned = re.sub(r'[ \-/]', '', m)

        # skip if it collapsed to fewer than 5 chars
        if len(cleaned) < 5:
            continue

        # skip if it’s all “1”s
        if all(ch == '1' for ch in cleaned):
            continue

        # skip explicit blacklisted codes
        if cleaned in blacklist:
            continue

        # count digits vs letters
        digits = sum(ch.isdigit() for ch in cleaned)
        letters = sum(ch.isalpha()  for ch in cleaned)
        # only allow if there are strictly more digits than letters
        if digits <= letters:
            continue

        results.append(cleaned)

    return results


def predict(pdf_path: str) -> List[int]:
    """
    Given the file path to a PDF, count its pages and return a random
    0/1 prediction for each page.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A list of integers (0 or 1), length == number of pages in the PDF.
    """
    pages = pdf_to_pages(pdf_path)

    result = [1]
    last_page_numbers = set(get_numbers(pages[0]))
    for page in pages[1:]:
        page_numbers = set(get_numbers(page))
        if len(page_numbers.intersection(last_page_numbers)) > 0:
            result.append(0)
        else:
            result.append(1)
            last_page_numbers = page_numbers
    return result
