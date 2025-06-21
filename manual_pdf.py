import glob, re,json
import pymupdf
from typing import Union
from typing import List
from tqdm import tqdm

def getmycities(s):
    top_20_german_freight_cities = [
    "Hamburg",
    "Bremerhaven",
    "Duisburg",
    "Wilhelmshaven",
    "Rostock",
    "Lübeck",
    "Emden",
    "Kiel",
    "Mannheim",
    "Frankfurt am Main",
    "Hannover",
    "Münster",
    "Stuttgart",
    "München",
    "Köln",
    "Dortmund",
    "Dresden",
    "Magdeburg",
    "Nürnberg",
    "Regensburg"
]
    return [s +" "+ i for i in top_20_german_freight_cities]
def timecheck(txt,main_page_types):
    # for i, line in enumerate(txt.splitlines()):
    #     print(line)
    #     line = line.decode("utf-8")  # Convert bytes to str
    t_match = re.search(txt,main_page_types["Time"][1])
    if t_match:
        return True
    return False

def type_match_check(line: str, type_str: Union[str, list[str]]) -> bool:
    if isinstance(type_str, str):
        return re.search(type_str, line) is not None
    elif isinstance(type_str, list):
        for pattern in type_str:
            if re.search(pattern, line):
                return True
        return False
    else:
        raise TypeError(f"Expected str or list[str], got {type(type_str)}")

def compare_Raul():
    json_files = glob.glob("labels.json")  # or use a specific folder path

    data_list = []
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            data_list.append(data)
    sum = 0
    fail=[]
    for i in tqdm(range(len(data_list[0]))):
        data = data_list[0][i]
        totest = manual_readPDF("data/BECONEX_challenge_materials_samples"+data["path"][4:])
        if data["label"] == totest[data["page"]]:
            sum +=1
        else:
            fail.append(i)
            # print('success! %d / %d '%(sum,i+1))
    print("Failed at:",fail)
    return 100 * sum / len(data_list[0])

def type_check(txt,type2check,main_page_types,line_lim):
    forcepass = False
    t_match_name = False

    # if main_page_types[type2check][2] == False:
    #     match_type_func = type_match_check
    # else:
    #     match_type_func = main_page_types[type2check][2]
    #     forcepass = True
    #     t_match_name = True
    match_type_func = type_match_check
    if main_page_types[type2check][2]:
        forcepass = True
        t_match_name = True
    # else:
        # match_type_func = main_page_types[type2check][2]

    for i in range(line_lim):
        # print(i)
        line = txt.splitlines()[i]
        line = line.decode("utf-8")  # Convert bytes to str        
        # first checking if there is the keyword on the line...
        if not forcepass:
            for j in main_page_types[type2check][0]:
            # print(type2check,j)
                t_match_name = match_type_func(line,j)
                if t_match_name:
                    # print("Match at %s using %s"%(type2check,j))
                    break
        if t_match_name:
            # print("%s found, now looking for more..."%type2check)
            ## check if there is information that matches this info
            if i+3 < len(txt.splitlines()):
                for i2 in range(3):
                    line2 = txt.splitlines()[i+i2]
                    line2 = line2.decode("utf-8")  # Convert bytes to str
                    t_match2 = match_type_func(line2,main_page_types[type2check][1])
                    if t_match2:
                        return True
    return False

def isthismainpage(page,main_page_types,line_lim=30):
    page_found = []
    text = page.get_text().encode("utf8") # get plain text (is in UTF-8)
    lines = text.splitlines()
    num_lines = len(lines)

    if line_lim ==None or line_lim> num_lines:
        line_lim = num_lines

    for type_it in (main_page_types):
        page_found.append(type_check(text,type_it,main_page_types,line_lim))

    if len(page_found)==0:
        print("no main page found...")
    return page_found


def manual_readPDF(pdf_path: str)-> List[int]:
    assert pdf_path != ""

    doc = pymupdf.open(pdf_path) # open a document
    main_page = []
    main_page_b = False
    pages_test = []
    lines2check = 80
    for page in doc: # iterate the document pages
        main_page = isthismainpage(page,main_page_types,lines2check)
        # print(main_page)
        if any(main_page):
            main_page_b = True
        else:
            main_page_b = False
        pages_test.append(main_page_b)
    
    pages_test=[int(b) for b in pages_test]
    # print(pages_test)
    return pages_test

def manual_4training_readPDF(pdf_path: str)-> List[int]:
    assert pdf_path != ""

    doc = pymupdf.open(pdf_path) # open a document
    main_page = []
    main_page_b = False
    pages_test = []
    NN_data = []
    lines2check = 80
    for page in doc: # iterate the document pages
        main_page = isthismainpage(page,main_page_types,lines2check)
        # print(main_page)
        if any(main_page):
            main_page_b = True
        else:
            main_page_b = False
        pages_test.append(main_page_b)
        NN_data.append(main_page)
    pages_test=[int(b) for b in pages_test]
    # print(pages_test)
    return pages_test,NN_data



Bestell = ["Bestellnr.:","BestellNr.:", "Bestell.:"]
Seite = [ "Seite 1", "Seite"]
Seite_str = [ "Seite 1", "1"]
Address = ["Rechnungsadresse","Address","addresse"]
Lieferschein = ["Lieferschein-Nr.", "Lieferschein"]
strasse = ["strasse","Str","Str.","str.","StraBe"]
telefon = ["Tel. :","Tel.:","Tel.: +49"]
Datum = ["Datum:", "Datum","Lieferdatum"]
Datum_str = [r'\b\d{2}.\d{2}.\d{4}\b',
                r'\b\d{2}.\d{2}.\d{2}\b',
                r'\b\d{2}/\d{2}/\d{4}\b',
                r'\b\d{2}/\d{2}/\d{2}\b']
Zip_code = getmycities(r'\b\d{2}.\d{2}.\d{4}\b')
cities = getmycities("")
Time = ["Time", "Time: "]
Time_str = [r'\b\d{4}\b']
main_page_types = {
                    "Time": [["Time"], Time_str,True],
                   "Datum": [Datum, Datum_str,False],
                    "Lieferschein":[Lieferschein,Lieferschein,True],
                    "Seite": [Seite,Seite,True],
                    "Bestell": [Bestell,Bestell,True],
                    "Address": [Address,Address,True],
                    "Zip_code":[Zip_code,Zip_code,True],
                    "Cities":[cities,cities,True],
                    "Telefone":[telefon,telefon,True],
                    "Rechnungsnummer": [Address,Address,True],
                    "strasse": [strasse,strasse,True],
                    "Address": [Address,Address,True],                    
                   }

if __name__ == "__main__":
    papath = "data/BECONEX_challenge_materials_samples/"
    pdfs = glob.glob(papath+"*.pdf")
    pdftouse = pdfs[0]
    totest = manual_4training_readPDF(pdftouse)

    print("The results using '%s' is: "%pdftouse)
    # print(totest[0])
    # print(totest[1])

    print("The score wrt to Raul's list: {:.2f} %".format(compare_Raul()))

    # for tt in pdf_files:
    #     totest = manual_readPDF(tt)
    #     print("The results using '%s' is: "%tt)
    #     print(totest)