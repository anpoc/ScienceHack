#%%
import json
import torch
import os

from typing import cast
from zipfile import ZipFile
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
from peft import LoraConfig

from colpali_engine.models import ColQwen2, ColQwen2Processor, \
    ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from transformers.models.qwen2_vl import Qwen2VLProcessor

#from colpaliRAG import ColQwen2ForRAG


def read_zip(args:dict):
    with ZipFile(args['base_path'], 'r') as zf:
        with zf.open(args['file_name'], 'r') as f:
            images = convert_from_bytes(f.read())
    return images


def read_folder(args:dict):
    images = convert_from_path(args['base_path'] + args['file_name'])
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
    return model, processor


def enable_retrieval(model) -> None:
    """
    Switch to retrieval mode.
    """
    model.enable_adapters()
    model._is_retrieval_enabled = True


def enable_generation(model) -> None:
    """
    Switch to generation mode.
    """
    model.disable_adapters()
    model._is_retrieval_enabled = False


if __name__ == "__main__":
    #os.chdir(path)
    device = get_torch_device("auto")
    cfg_path = "./src/colpali/cfg.json"
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    # Get images
    if cfg['data']['base_path'].endswith('zip'):
        images = read_zip(cfg['data'])
    else:
        images = read_folder(cfg['data'])
    
    # Get the LoRA config from the pretrained retrieval model
    model_name = cfg['model']['model_name']
    lora_config = LoraConfig.from_pretrained(model_name)
    # Load processor and model
    # model, processor = get_model(cfg['model']['model_name'])
    processor = Qwen2VLProcessor.from_pretrained(
        lora_config.base_model_name_or_path, use_fast=True)
    #model = ColQwen2.from_pretrained(
    #        model_name,
    #        torch_dtype=torch.bfloat16,
    #        device_map="auto",
    #        attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
    #).eval()
    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", device_map="auto").eval()

    # Inputs
    query = "Quelle partie de la production pétrolière du Kazakhstan provient de champs en mer ?"
    # Preprocess the inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                },
                {
                    "type": "text",
                    "text": f"Answer the following question using the input image: {query}",
                },
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    inputs_generation = processor(
        text=[text_prompt],
        images=[images[0]],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate the RAG response
    # enable_generation(model)
    output_ids = model.generate(**inputs_generation, max_new_tokens=128)
    #output_ids = Qwen2VLForConditionalGeneration(**inputs_generation, max_new_tokens=128)

    # Ensure that only the newly generated token IDs are retained from output_ids
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs_generation.input_ids, output_ids)]

    # Decode the RAG response
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print(output_text)