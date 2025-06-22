import os
import json
import argparse
from colpali.process import extract_info
from colpali.postprocess import postprocess
from matching_stage.chunk_to_customer_matching import match_page_to_customers
from nils_method.predict_splits import predict as predict_split


def chunk_starts(chunk_ids, split_pred):
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
        if split_pred[idx] == 0:
            continue
        if val == -1:
            continue
        if val != current_value:
            result.append((val, idx))
            current_value = val
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description="Process PDF files using regex-based splitting and matching",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
    Examples:
    python main_script.py data/sample.pdf data/vendors.json
    python main_script.py -o results.json data/sample.pdf data/vendors.json
    """
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the input PDF file"
    )
    
    parser.add_argument(
        "json_path", 
        type=str,
        help="Path to the vendor data JSON file"
    )

    
    args = parser.parse_args()

    file_name = args.pdf_path.split('/')[-1].split('.')[0]
    
    os.chdir('/workspaces/ScienceHack/')
    cfg_path = './src/cfg.json'

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    with open(args.json_path, 'r') as f:
        sap_records = json.load(f)

    if cfg['model']['reprocess_flag'] or not os.path.exists(
        f"{cfg['results']['save_path']}cl_results_{file_name}.json"
    ):
        extract_info(cfg, args.pdf_path)
        postprocess(cfg, args.pdf_path)
    
    
    chunk_ids = match_page_to_customers(
        args.pdf_path, 
        args.json_path,
        f"{cfg['results']['save_path']}cl_results_{file_name}.json"
    ).tolist()

    split_pred = predict_split(args.pdf_path)
        
    final_pred = []
    for record_pos, pg_pos in chunk_starts(chunk_ids, split_pred):
        final_pred.append({
            'page': pg_pos,
            'MBLNR': sap_records[record_pos]['MBLNR'],
            'MJAHR': sap_records[record_pos]['MJAHR']
        })
        
    with open(f"{cfg['results']['save_path']}results_{file_name}.json", 'w') as f:
        json.dump(final_pred, f)