import os, json

from collections import Counter
from dateutil import parser


def standardize_date(date_str):
    """
    Converts various date formats to 'YYYY-MM-DD' format.
    
    Parameters:
        date_str (str): The input date string.
    
    Returns:
        str: The date in 'YYYY-MM-DD' format.
    
    Raises:
        ValueError: If the date string cannot be parsed.
    """
    try:
        if date_str == '29. Juni 2017':
            date_str = '29. June 2017'
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime("%Y-%m-%d")
    except (ValueError, TypeError) as e:
        print(f"Invalid date format: {date_str}")
        return date_str
    

def extract_after_keyword(text, keywords:str=[' is ', ' issued on ', ' was dated ']):
    """
    Extracts the part of the string that comes after ' is ',
    strips whitespace, and removes a trailing period if present.
    
    Parameters:
        text (str): The input string.
    
    Returns:
        str: The cleaned substring after ' is '.
    """
    for keyword in keywords:
        if 'not found' in text or 'not provided' in text:
            return 'NA'
        if keyword in text:
            result = text.split(keyword, 1)[1].strip()
            if result.endswith('.'):
                result = result[:-1].rstrip()
            return result
    return text



def get_mode_or_first(list_str:list, exclude:str=''):
    """
    Returns the string that occurs more than half the time (majority element).
    If no majority exists, returns the first element.
    If the list is empty, returns None.
    
    Parameters:
        list_str (list of str): The input list.
    
    Returns:
        str or None: The majority string, or first string if no majority, or None.
    """
    if exclude in list_str:
        list_str.remove(exclude) 
    if not list_str:
        return 'NA'
    most_common = Counter(list_str).most_common(1)
    if most_common[0][1] > 1 or len(list_str) == 1:
        return most_common[0][0]
    else:
        if 'NA' in list_str:
            list_str.remove('NA') 
        return list_str if len(list_str) > 1 else list_str[0]


def postprocess(cfg:dict, pdf_file:str):
    file_name = pdf_file.split('/')[-1].split('.')[0]
    with open(f"{cfg['results']['save_path']}tmp_results_{file_name.split('/')[-1]}.json", 'r') as f:
        outputs = json.load(f)
    output_dict = {}
    for k, v in outputs.items():
        page_dict = {
            'MJAHR': None,
            'Delivery Note Number': None,
            'Delivery Note Date': None,
            'Purchase Order Number': None,
            'Vendor - Name 1': None,
            'Vendor - Address': None
        }
        qnumber = 0
        field_aux = []
        dn_num = ''
        for kv, vv in v.items():
            if 1 < qnumber < 4:
                field_aux.append(
                    standardize_date(extract_after_keyword(str(vv[0])))
                )
            else:
                field_aux.append(extract_after_keyword(str(vv[0])))

            if qnumber == 1:
                dn_num = get_mode_or_first(field_aux)
                page_dict['Delivery Note Number'] = dn_num
                field_aux = []
            elif qnumber == 3:
                page_dict['Delivery Note Date'] =\
                    get_mode_or_first(field_aux)
                if isinstance(page_dict['Delivery Note Date'], list):
                    page_dict['MJAHR'] = get_mode_or_first([
                        x.split('-')[0] for x in page_dict['Delivery Note Date']
                    ])
                else:
                    page_dict['MJAHR'] =\
                        page_dict['Delivery Note Date'].split('-')[0]
                field_aux = []
            elif qnumber == 7:
                page_dict['Vendor - Name 1'] = get_mode_or_first([
                    'NA' if 'beconex' in x.lower() else x for x in field_aux 
                ])
                field_aux = []
            elif qnumber == 9:
                page_dict['Vendor - Address'] =\
                    get_mode_or_first(field_aux)
                field_aux = []
            elif qnumber == 11:
                page_dict['Purchase Order Number'] =\
                    get_mode_or_first(field_aux, exclude=dn_num)
                field_aux = []
            qnumber += 1
        output_dict[k] = page_dict
    with open(f"{cfg['results']['save_path']}cl_results_{file_name.split('/')[-1]}.json", 'w') as f:
        json.dump(output_dict, f)


if __name__ == "__main__":
    os.chdir('/home/hackaton2025/ScienceHack/')

    cfg_path = './src/cfg.json'
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)
    
    for file_name in cfg['data']['file_name']:
        postprocess(cfg, pdf_file=file_name)