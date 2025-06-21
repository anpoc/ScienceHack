import os
import json


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
    with open('./data/SAP_data.json', 'r') as f:
        sap_records = json.load(f)

    # open config and modify file and path
    # call the colpali main
    # call the colpali postprocess
    # call benjis file

    chunk_ids = []

    final_pred = []
    for record_pos, pg_pos in chunk_starts(chunk_ids):
        final_pred.append({
            'page': pg_pos,
            'MBLNR': sap_records[record_pos]['MBLNR'],
            'MJAHR': sap_records[record_pos]['MJAHR']
        })
    with open('./results/example.json', 'w') as f:
        json.dump(final_pred, f)