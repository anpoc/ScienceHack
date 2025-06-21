import random
from typing import List
import fitz
import torch
from evaluation import evaluate
from src.text_based_split.utils.utils import pdf_to_embeds
from src.text_based_split.model import Classifier

def predict(pdf_path: str, model = None) -> List[int]:
    """
    Given the file path to a PDF, count its pages and return a random
    0/1 prediction for each page.

    Args:
        pdf_path: Path to the input PDF file.

    Returns:
        A list of integers (0 or 1), length == number of pages in the PDF.
    """
    if model == None:
        model = Classifier()
        model.load_state_dict(torch.load("./src/text_based_split/ckpts/model_last.pt"))
    model.eval()
    for param in model.parameters():
            param.requires_grad = False
    
    # Load the PDF with pypdf
    pdf = fitz.open(pdf_path)
    num_pages = len(pdf)
    file2embeds=pdf_to_embeds(pdf_path)
    embeddings_batch = []
    for idx in range(num_pages):
        embeddings_batch.append(file2embeds.callback(idx))
    embeddings_batch = torch.stack(embeddings_batch).squeeze(dim=1)
    outputs = model(torch.tensor(embeddings_batch))
    preds = (outputs >= 0.5).float()

    # Randomly choose 0 or 1 for each page
    return preds.tolist()

if __name__ == "__main__":
    accuracy, chunk_score = evaluate(predict, split="test", n=10)