#%%
import os, json, torch

from tqdm import tqdm
from typing import cast
from zipfile import ZipFile
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image

from colpali_engine.utils.torch_utils import get_torch_device
from peft import LoraConfig
from transformers.utils.import_utils import is_flash_attn_2_available
from transformers.models.qwen2_vl import Qwen2VLProcessor

from colpali.colpaliRAG import ColQwen2ForRAG


def read_zip(base_path:str, file_name:str):
    with ZipFile(base_path, 'r') as zf:
        with zf.open(file_name, 'r') as f:
            images = convert_from_bytes(f.read())
    return images


def read_file(pdf_file):
    images = convert_from_path(pdf_file)
    return images


def scale_image(image: Image.Image, new_height:int=1024) -> Image.Image:
    """
    Scale an image to a new height while maintaining the aspect ratio.
    """
    width, height = image.size
    aspect_ratio = width / height
    new_width = int(new_height * aspect_ratio)

    scaled_image = image.resize((new_width, new_height))

    return scaled_image


def extract_info(cfg, pdf_file:str):
    device = get_torch_device('auto')

    # Get the LoRA config from the pretrained retrieval model
    model_name = cfg['model']['model_name']
    lora_config = LoraConfig.from_pretrained(model_name)
    # Load the processors
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    processor = cast(
        Qwen2VLProcessor, 
        Qwen2VLProcessor.from_pretrained(
            lora_config.base_model_name_or_path, use_fast=True
        )
    )
    # Load the model with the loaded pre-trained adapter for retrieval
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = cast(
        ColQwen2ForRAG,
        ColQwen2ForRAG.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2" if is_flash_attn_2_available() else "sdpa",
        ),
    )

    file_name = pdf_file.split('/')[-1].split('.')[0]
    images = read_file(pdf_file)
    
    # Preprocess the inputs
    output_dict = {}
    for id_img, img in enumerate(images):
        output_dict[id_img] = {}
        for query in tqdm(cfg['model']['text_queries']):
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
                images=[img],
                padding=True,
                return_tensors="pt",
            ).to(device)

            # Generate the RAG response
            model.enable_generation()
            output_ids = model.generate(
                **inputs_generation, max_new_tokens=128
            )

            # Ensure that only the newly generated token IDs are retained from output_ids
            generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs_generation.input_ids, output_ids)]

            # Decode the RAG response
            output_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            output_dict[id_img][query] = output_text
    
    with open(f"{cfg['results']['save_path']}tmp_results_{file_name.split('/')[-1]}.json", 'w') as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    os.chdir('/home/hackaton2025/ScienceHack/')
    device = get_torch_device('auto')

    cfg_path = './src/cfg.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    extract_info(cfg)