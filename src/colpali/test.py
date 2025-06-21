# pip install colpali-engine # from PyPi
# pip install git+https://github.com/illuin-tech/colpali # from source
# !pip install --no-deps fast-plaid fastkmeans

# Models to try: vidore/colSmol-500M

#%%
import torch
import gc
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor, \
    ColPali, ColPaliProcessor

#%%
gc.collect()
torch.cuda.empty_cache()
model_name =  "vidore/colpali-v1.3"# ["vidore/colpali-v1.3", "vidore/colqwen2.5-v0.2"]
model = ColPali.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda").eval()
processor = ColPaliProcessor.from_pretrained(model_name, use_fast=True)

#%%
if 'colqpali' in model_name:
    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    ).eval()
    processor = ColPaliProcessor.from_pretrained(model_name, use_fast=True)
else:
    model = ColQwen2.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
    ).eval()
    processor = ColQwen2Processor.from_pretrained(model_name, use_fast=True)

#%%

# Your inputs
# TBD
images = [
    Image.new("RGB", (128, 128), color="white"),
    Image.new("RGB", (64, 32), color="black"),
]
queries = [
    "Is this the start of a delivery note?",
    "What is the delivery note number?",
]

# Process the inputs
batch_images = processor.process_images(images).to(model.device)
batch_queries = processor.process_queries(queries).to(model.device)

# Forward pass
with torch.no_grad():
    image_embeddings = model(**batch_images)
    query_embeddings = model(**batch_queries)

scores = processor.score_multi_vector(query_embeddings, image_embeddings)