from transformers import pipeline
import fitz

ner = pipeline("ner", grouped_entities=True, model="Davlan/bert-base-multilingual-cased-ner-hrl")


doc = fitz.open("/workspaces/ScienceHack/data/challenge/20170303_Holtz Office Support/0001.pdf")[0]
passage =doc.get_text()
        
results = ner(passage)
for r in results:
    print(r)