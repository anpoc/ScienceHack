from typing import List
import fitz
import re


def page_has_seite_marker(text: str) -> bool:
    # match "Seite", optional spaces, digit 2-9, lookahead for space or slash
    pattern = re.compile(r"Seite\s*[2-9](?=\s|/)")
    return bool(pattern.search(text))


def predict(pdf_path: str) -> List[int]:
    """
    Processes each page of the PDF at `filename`.
    Returns:
      last_texts: List of the last text block (str) on each page.
      flags:      List of ints (0 if last character is 2–9, else 1).
    """
    doc = fitz.open(pdf_path)
    last_texts = []
    flags = []

    for page in doc:
        pred = 1
        blocks = page.get_text("blocks")
        if not blocks:
            last_texts.append("")
            flags.append(1)
            continue

        last_block = max(blocks, key=lambda b: b[3])
        text = last_block[4]
        stripped = text.rstrip()  # remove trailing whitespace/newlines

        last_texts.append(stripped)

        if len(stripped) == 1 and stripped[-1] in "23456789":
            pred = 0
        full_text = page.get_text("text")
        if page_has_seite_marker(full_text):
            pred = 0
        flags.append(pred)
    doc.close()
    return flags
