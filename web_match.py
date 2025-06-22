from typing import List, Set
import fitz
import re


def pdf_to_pages(path: str) -> List[str]:
    """Load a PDF and return a list of its pages’ text."""
    pdf = fitz.open(path)
    return [page.get_text() for page in pdf]


def get_contacts(text: str) -> List[str]:
    """
    Find all e-mail addresses and URLs in a block of text.
    Returns a list of the raw matches.
    """
    email_regex = re.compile(
        r'(?i)\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    )
    url_regex = re.compile(
        r'(?i)\b(?:https?://|www\.)\S+\b'
    )
    emails = email_regex.findall(text)
    urls = url_regex.findall(text)
    return emails + urls


def predict(pdf_path: str) -> List[int]:
    """
    Given a PDF path, produce a list of 0/1 flags—one per page—
    where 1 means “new contacts” appeared on that page, and 0 means none
    that weren’t already on the previous page.
    """
    pages = pdf_to_pages(pdf_path)

    # Initialize with page 0 always flagged as “new”
    result: List[int] = [1]
    last_seen: Set[str] = set(get_contacts(pages[0]))

    for page_text in pages[1:]:
        current = set(get_contacts(page_text))
        # If there's any overlap, no new “section”
        if current & last_seen:
            result.append(0)
        else:
            result.append(1)
            last_seen = current

    return result
