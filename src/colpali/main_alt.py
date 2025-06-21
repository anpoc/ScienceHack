from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from pdf2image import convert_from_path
import groq

RAG = RAGMultiModalModel.from_pretrained("vidore/colpali")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", 
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="cuda"
)

RAG.index(input_path="Data/input.pdf",
          index_name="multimodal_rag",
          store_collection_with_index=False,
          overwrite=True,)

text_query = "What is the type of star hosting thge kepler-51 planetary system?"
results = RAG.search(text_query,k=3)
results

###### RESPOSNE #####
[{'doc_id': 1, 'page_num': 1, 'score': 25.125, 'metadata': {}, 'base64': None},
 {'doc_id': 1, 'page_num': 8, 'score': 24.875, 'metadata': {}, 'base64': None},
 {'doc_id': 1, 'page_num': 9, 'score': 24.125, 'metadata': {}, 'base64': None}]

images = convert_from_path("Data/input.pdf")
image_index = results[0]["page_num"] -1
