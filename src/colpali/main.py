#%%
import argparse
import json
import torch
import gc

from zipfile import ZipFile
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available

from colpali_engine.models import ColQwen2, ColQwen2Processor, \
    ColPali, ColPaliProcessor

#%%
def read_zip(args:dict, poppler_path:str):
    with ZipFile(args['base_path'], 'r') as zf:
        with zf.open(args['file_name'], 'r') as f:
            images = convert_from_bytes(f.read(), poppler_path=poppler_path)
    return images


def read_folder(args:dict, poppler_path:str):
    images = convert_from_path(args['base_path'] + args['file_name'], poppler_path=poppler_path)
    return images


def get_model(model_name:str):
    if 'colqpali' in model_name:
        model = ColPali.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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


if __name__ == "__main__":
    #os.chdir(path)
    cfg_path = "./cfg.json"
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    # Get images
    if cfg['data']['base_path'].endswith('zip'):
        images = read_zip(cfg['data'], cfg['extras']['poppler'])
    else:
        images = read_folder(cfg['data'], cfg['extras']['poppler'])
    
    # Get the model and processor 
    model, processor = get_model(cfg['model']['model_name'])

    # Process the inputs
    batch_images = processor.process_images(images).to(model.device)
    batch_queries = processor.process_queries(cfg['model']['text_queries'])\
        .to(model.device)
    with torch.no_grad():
        image_embeddings = model(**batch_images)
        query_embeddings = model(**batch_queries)
    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    