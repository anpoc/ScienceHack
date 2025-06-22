import fitz
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
class pdf_to_embeds():
    def __init__(self, pdf_path):
        self.pdf = fitz.open(pdf_path)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
         
    def tokenize_text(self, text:str):
            batch_dict = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
            return batch_dict
        
    def embedd_text(self, tokens):
        outputs = self.model(**tokens)
        embeddings = self.average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
            
        return embeddings

    def get_page(self, idx):
        page = self.pdf[idx]
        passage = page.get_text()
        text = [f"passage: {passage}"]
        return text
    
    def callback(self, idx):
         text = self.get_page(idx)
         tokens = self.tokenize_text(text)
         embeddings = self.embedd_text(tokens)
         return embeddings
