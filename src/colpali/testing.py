#%%
import argparse
import json

from zipfile import ZipFile
from pdf2image import convert_from_path, convert_from_bytes

#%%
def read_zip(args:dict, poppler_path:str):
    with ZipFile(args['base_path'], 'r') as zf:
        with zf.open(args['file_name'], 'r') as f:
            images = convert_from_bytes(f.read(), poppler_path=poppler_path)
    return images


def read_folder(args:dict, poppler_path:str):
    images = convert_from_path(args['base_path'] + args['file_name'], poppler_path=poppler_path)
    return images


if __name__ == "__main__":
    #os.chdir(path)
    cfg_path = "./cfg.json"
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    if cfg['data']['base_path'].endswith('zip'):
        imgs = read_zip(cfg['data'], cfg['extras']['poppler'])
    else:
        imgs = read_folder(cfg['data'], cfg['extras']['poppler'])
    