from typing import List
import fitz
import torch
from .model import Classifier
from .utils.utils import pdf_to_embeds


def predict(pdf_path: str, model = None) -> List[int]:
    if model == None:
        model = Classifier()
        model.load_state_dict(torch.load("./src/text_based_split/ckpts/model_best_acc.pt"))
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
    #preds = (outputs >= 0.5).float()

    return outputs

