import json
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import fitz

class JSONDataset(Dataset):
    def __init__(self, sap_data="data/SAP_data.json", labels="/workspaces/ScienceHack/src/text-based-split/labels.json"):
        with open(sap_data, 'r', encoding='utf-8') as f:
            self.sap_data = json.load(f)
        with open(labels, 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def __len__(self):
        return len(self.labels)

    
    def tokenize_text(self, text:str):
        batch_dict = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
        return batch_dict
    
    def embedd_text(self, tokens):
        outputs = self.model(**tokens)
        embeddings = self.average_pool(outputs.last_hidden_state, tokens['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


    def get_page(self,label_dct: dict):
        pdf_path = label_dct["path"]
        pdf_page = label_dct["page"]
        doc = fitz.open(pdf_path)[pdf_page]
        passage =doc.get_text()
        text = [f"passage: {passage}"]
        return text



    def __getitem__(self, idx):
        text = self.get_page(self.labels[idx])
        label = self.labels[idx]["label"]
        tokenized_text = self.tokenize_text(text)
        embeddings = self.embedd_text(tokenized_text)

        return embeddings, label

# Example usage


if __name__ == "__main__":
    dataset = JSONDataset()
    dataset[1]