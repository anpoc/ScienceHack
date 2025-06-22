import os
import json

from colpali.process import extract_info
from colpali.postprocess import postprocess
from matching_stage.chunk_to_customer_matching import match_page_to_customers


def chunk_starts(chunk_ids):
    """
    Returns a list of (value, index) for each new chunk start,
    triggered when the number changes (ignoring -1s).
    
    Parameters:
        chunk_ids (list of int): Input list.
    
    Returns:
        list of tuples: (value, index) for chunk starts.
    """
    if not chunk_ids:
        return []

    result = []
    current_value = chunk_ids[0] if chunk_ids[0] != -1 else None
    result.append((current_value, 0))
    for idx in range(1, len(chunk_ids)):
        val = chunk_ids[idx]
        if val == -1:
            continue
        if val != current_value:
            result.append((val, idx))
            current_value = val
    return result


if __name__ == "__main__":
    os.chdir('/home/hackaton2025/ScienceHack/')
    cfg_path = './src/cfg.json'

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    with open(f"{cfg['data']['base_path']}{cfg['data']['records_file']}", 'r') as f:
        sap_records = json.load(f)

    if cfg['model']['reprocess_flag'] or not os.path.exists(
        f"{cfg['results']['save_path']}cl_results_{cfg['data']['file_name'][0].split('/')[-1]}.json"
    ):
        extract_info(cfg)
        postprocess(cfg)
    
    for file_name in cfg['data']['file_name']:
        chunk_ids = match_page_to_customers(
            f'{cfg['data']['base_path']}{file_name}.pdf', 
            f'{cfg['data']['base_path']}{cfg['data']['records_file']}',
            f"{cfg['results']['save_path']}cl_results_{file_name.split('/')[-1]}.json"
        ).tolist()

        final_pred = []
        for record_pos, pg_pos in chunk_starts(chunk_ids):
            final_pred.append({
                'page': pg_pos,
                'MBLNR': sap_records[record_pos]['MBLNR'],
                'MJAHR': sap_records[record_pos]['MJAHR']
            })
        
        with open(f'{cfg['results']['save_path']}results_{file_name.split('/')[-1]}.json', 'w') as f:
            json.dump(final_pred, f)