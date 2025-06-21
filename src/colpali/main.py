#%%
import json
import torch
import os

from typing import cast
from zipfile import ZipFile
from pdf2image import convert_from_path, convert_from_bytes
from transformers.utils.import_utils import is_flash_attn_2_available
from peft import LoraConfig

from PIL import Image

from colpali_engine.models import ColQwen2, ColQwen2Processor, \
    ColPali, ColPaliProcessor
from colpali_engine.utils.torch_utils import get_torch_device
from transformers.models.qwen2_vl import Qwen2VLProcessor

from colpaliRAG import ColQwen2ForRAG


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


def scale_image(image: Image.Image, new_height: int = 1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image


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
    print(images[7].size)
    print(images[8].size)
    
    # Get the LoRA config from the pretrained retrieval model
    model_name = cfg['model']['model_name']
    lora_config = LoraConfig.from_pretrained(model_name)
    # Load processor and model
    # model, processor = get_model(cfg['model']['model_name'])
    # Load the processors
    processor = cast(Qwen2VLProcessor, Qwen2VLProcessor.from_pretrained(lora_config.base_model_name_or_path))

    # Load the model with the loaded pre-trained adapter for retrieval
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = cast(
        ColQwen2ForRAG,
        ColQwen2ForRAG.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
        ),
    )

    # Inputs
    #query = "Who is the vendor of this delivery note?"
    #query = "What is the date of this delivery note?"
    query = "What is the order number or delivery note identifier?"
    # query = "Are the words delivery note or Lieferschein in this document?"
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
        images=[scale_image(img, new_height=512) for img in images[7:8]],
        padding=True,
        return_tensors="pt",
    ).to(device)

    order number
    delivery note number
    date
    name of vendor
    address of vendor

    # Generate the RAG response
    model.enable_generation()
    output_ids = model.generate(**inputs_generation, max_new_tokens=128)

    # Ensure that only the newly generated token IDs are retained from output_ids
    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs_generation.input_ids, output_ids)]

    # Decode the RAG response
    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    print(output_text)