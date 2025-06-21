import re
from pypdf import PdfReader

from evaluation import evaluate


def regex_split(pdf_file):

    # Open the PDF file
    text_per_page = []
    with open(pdf_file, "rb") as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text_per_page.append(page.extract_text())

    predicted = []

    previous_identifier_numbers = []

    for i, text in enumerate(text_per_page):
        if i == 0:  # the first page is always the start of a new document
            predicted.append(1)
            # print(f"Page {i}: First page must be a new document")
            continue

        # match patterns have hierarchical odering
        # page_pattern = r"Seite.*?\b(\d+)(?:\s*(?:von|/)\s*(\d+))?\b"
        # # lieferschein_pattern = r'(?i)\blieferschein\b'

        # # first try to match the page pattern
        # page_number_match = re.search(page_pattern, text, re.IGNORECASE | re.DOTALL)
        # if page_number_match:
        #     current_page = int(page_number_match.group(1))
        #     total_pages = page_number_match.group(2)
        #     if total_pages:
        #         total_pages = int(total_pages)

        #         if (
        #             current_page < 1000
        #         ):  # say we dont know if the page number is unreasonably high
        #             if current_page == 1:
        #                 predicted.append(1)
        #             else:
        #                 predicted.append(0)
        #             continue

        # secondly, try to match the Lieferschein pattern
        # lieferschein_match = re.search(lieferschein_pattern, text)
        # if lieferschein_match:
        #     # print(f"Page {i}: Found match – Lieferschein")
        #     predicted.append(1)
        #     continue
        # else:
        #     # print(f"Page {i}: No Lieferschein matches found.")
        #     pass

        # if no matches found, be transparent about it and append -1
        # print(f"Page {i}: No matches found, appending -1")

        # try to find an identifier number
        # if it matches the previous page, it is part of the same document
        # store it for the next page
        pattern = r"""
(?i)
\w*
(?:nr|nummer)
[.:/\-]*
\s*
(
  (?=[\d \-]*\d)       # must contain at least one digit
  [\d \-]*\d           # digits, spaces, dashes, ending with digit (no trailing space)
)
"""
        identifier_numbers = re.findall(pattern, text, re.VERBOSE)
        # if any identifier number matches the previous page, it is part of the same document
        if identifier_numbers:
            # print(f"Page {i}: Found identifier numbers: {identifier_numbers}")
            if (
                len(set(identifier_numbers).intersection(previous_identifier_numbers))
                > 0
            ):
                predicted.append(0)
                previous_identifier_numbers = identifier_numbers
                continue
            else:
                # if no identifier number matches the previous page, it is a new document
                predicted.append(1)
                previous_identifier_numbers = identifier_numbers
                continue
        else:
            # if no identifier number is found, just assume it is part of the same document
            predicted.append(0)
            previous_identifier_numbers = []

    # for evaluation purposes, turn -1 into 0
    return predicted


if __name__ == "__main__":
    evaluate(regex_split, split="test", n=100)

    # print(
    #     regex_split(
    #         "C:/Users/benja/sciencehackathon/BECONEX_challenge_materials_samples/batch_5_2022_2.pdf"
    #     )
    # )
