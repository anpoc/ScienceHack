from typing import List
import torch
from .model import Classifier
from .manual_pdf import *

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
        num_inputs = len(list(main_page_types.keys()))
        model = Classifier(input_dim=num_inputs, hidden_dim=num_inputs//2)
        model.load_state_dict(torch.load("./src/manuel/ckpts/model_best_acc.pt"))
    model.eval()
    for param in model.parameters():
            param.requires_grad = False
    
    # Load the PDF with pypdf
    tuple_lists = manual_4training_readPDF(pdf_path)

    list_parameters = tuple_lists[1]

    outputs = model(torch.tensor(list_parameters, dtype=torch.float))
    #preds = (outputs >= 0.5).float()

    # Randomly choose 0 or 1 for each page
    return outputs
if __name__ == "__main__":
    pass
    #exact_match, accuracy, chunk_score = evaluate(predict, split="test", n=100)