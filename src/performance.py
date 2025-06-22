import json, os


def start_identification(pred_dict, real_dict):
    pred_start = [x['page'] for x in pred_dict]
    real_start = [x['page'] for x in real_dict]
    TP = set(pred_start).intersection(set(real_start))
    FP = set(pred_start).difference(set(real_start))
    FN = set(real_start).difference(set(pred_start))
    return TP, FP, FN


def info_match(pred_dict, real_dict, TP):
    pred_info = dict([(x['page'], f"{x['MBLNR']}{x['MJAHR']}") for x in pred_dict if x['page'] in TP])
    real_info = dict([(x['page'], f"{x['MBLNR']}{x['MJAHR']}") for x in real_dict if x['page'] in TP])
    match = 0
    nomatch = 0
    for k in pred_info.keys():
        if pred_info[k] == real_info[k]:
            match += 1
        else:
            nomatch += 1   
    return match, nomatch


if __name__ == "__main__":
    os.chdir('/home/hackaton2025/ScienceHack/')
    cfg_path = './src/cfg.json'

    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    for file_name in cfg['data']['file_name']:
        with open(
            f"{cfg['data']['base_path']}output_{file_name}.json", 'r'
        ) as f:
            pred_dict = json.load(f)
        with open(
            f"{cfg['results']['save_path']}results_{file_name}.json", 'r'
        ) as f:
            real_dict = json.load(f)
        
        TP, FP, FN = start_identification(pred_dict, real_dict)
        match, nomatch = info_match(pred_dict, real_dict, TP)
        print(f'Results for {file_name}\n')
        print(f'-- Start detection performance: {len(TP)} (TP), {len(FP)} (FP), {len(FN)} (FN)\n')
        print(
            f'-- Delivery note info extraction: {round(match / (match + nomatch), 4) * 100}%\n'
        )