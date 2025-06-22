from typing import List
import fitz
import re
import dateparser


def pdf_to_pages(path: str) -> str:
    pdf = fitz.open(path)
    pages = []
    for page in pdf:
        pages.append(page.get_text())
    return pages

def get_dates(text: str) -> List[str]:
    pattern = r"""(?x)               # free-spacing mode
    (?<!\d)                        # no digit before
    (?:
        # 31-day months
        (?:
        (?:0?[1-9]|[12]\d)|3[01]
        )\s?[./:-][\s.]?
        (?:0?[13578]|1[02]
        |J(?:an(?:uar)?|uli?)
        |M(?:ärz?|ai)
        |Aug(?:ust)?
        |Okt(?:ober)?
        |Dez(?:ember)?
        )\s?(?:[./:-][\s.]?)?
        [1-9]\d\d\d

        | # 30-day months
        (?:
        (?:0?[1-9]|[12]\d)|30
        )\s?[./:-][\s.]?
        (?:0?[13-9]|1[012]
        |J(?:an(?:uar)?|u[nl]i?)
        |M(?:ärz?|ai)
        |A(?:pr(?:il)?|ug(?:ust)?)
        |Sep(?:tember)?
        |Okt(?:ober)?
        |(?:Nov|Dez)(?:ember)?
        )\s?(?:[./:-][\s.]?)?
        [1-9]\d\d\d

        | # February 29 on leap years
        (?:0?[1-9]|[12]\d)\s?[./:-][\s.]?
        (?:0?2|Fe(?:b(?:ruar)?)?)\s?(?:[./:-][\s.]?)?
        [1-9]\d
        (?:[02468][048]|[13579][26])

        | # February other days
        (?:0?[1-9]|[12][0-8])\s?[./:-][\s.]?
        (?:0?2|Fe(?:b(?:ruar)?)?)\s?(?:[./:-][\s.]?)?
        [1-9]\d\d\d
    )
    (?!\d)                        # no digit after
    """
    date_regex = re.compile(pattern, re.VERBOSE | re.IGNORECASE)
    matches = date_regex.findall(text)
    dates = []
    for d in matches:
        if not d:
            continue
        parsed = dateparser.parse(d, languages=['de'])
        if parsed:
            dates.append(parsed.strftime('%Y-%m-%d'))
    return dates


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
    pages = pdf_to_pages(pdf_path)

    result = [1]
    last_page_dates = set(get_dates(pages[0]))
    for page in pages[1:]:
        page_dates = set(get_dates(page))
        if len(page_dates.intersection(last_page_dates)) > 0:
            result.append(0)
        else:
            result.append(1)
            last_page_dates = page_dates
    return result
