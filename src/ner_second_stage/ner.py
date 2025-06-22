from transformers import pipeline
import fitz
import json
def process_page(ner, passage, data):
    results = ner(passage)

    max_counts = 0
    ids = []
    for idx, dct in enumerate(data):
        vendor = dct['Vendor - Name 1']
        city = dct["Vendor - Address - City"]
        country = dct["Vendor - Address - Country"]
        street = dct["Vendor - Address - Street"]
        count = 0
        for r in results:
            if r["word"] in vendor:
                count+=1
            if count==1:
                if city is not None and r["word"] in city:
                    count+=1
                if country is not None and  r["word"] in country:
                    count+=1
                if street is not None and r["word"] in street:
                    count+=1
        if count>max_counts:
            max_counts = count
            ids=[idx]
        elif count == max_counts and count!=0:
            ids.append(idx)

    if len(ids) == 1:
        return ids[0]
    elif len(ids) == 0:
        return -1
    else:
        for i,id in enumerate(ids):

            a = str(data[id]["Purchase Order Number"])
            b = str(data[id]["Delivery Note Number"])

            if a in passage or b in passage:
                return id
        return -1

def dict_generator(sap_data, pred, page=-1):
    dct = {}
    dct["page"] = page
    if pred ==-1:
        dct["MBLNR"] = None
        dct["MJAHR"] = None
    else:
        dct["MBLNR"] = sap_data[pred]["MBLNR"]
        dct["MJAHR"] = sap_data[pred]["MJAHR"]
    return dct
    

def data_processor(sap_data_path, pdf_path, preds_stage_one):
    ner = pipeline("ner", grouped_entities=True, model="Davlan/bert-base-multilingual-cased-ner-hrl")
    with open(sap_data_path) as json_data:
            sap_data = json.load(json_data)
    doc = fitz.open(pdf_path)
    json_list = []
    for idx, pred_so in enumerate(preds_stage_one):
        if pred_so == 0:
            continue
        page = doc[idx]
        passage = page.get_text()
        pred = process_page(ner, passage, sap_data)
        if pred != -1:
            json_list.append(dict_generator(sap_data, pred, idx))
    with open("out.json", "w") as json_file:
        json.dump(json_list, json_file, indent=4)
        
        


if __name__ =="__main__":
    data_processor("./src/ner_second_stage/sap_data.json", 
                   "/workspaces/ScienceHack/data/BECONEX_challenge_materials_samples/batch_1_2017_2018.pdf",
                   [1,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1])