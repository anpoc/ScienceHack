import glob, re
import pymupdf
from typing import Union
from typing import List



Bestell = ["Bestellnr.:","BestellNr.:", "Bestell.:"]
Seite = [ "Seite 1", "Seite"]

Datum = ["Datum:", "Datum","Lieferdatum"]
Datum_str = [r'\b\d{2}.\d{2}.\d{4}\b',
                r'\b\d{2}.\d{2}.\d{2}\b',
                r'\b\d{2}/\d{2}/\d{4}\b',
                r'\b\d{2}/\d{2}/\d{2}\b']

Time = ["Uhr: ", "Time", "Time: "]
main_page_types = {"Bestell": [Bestell,Bestell,None],
                   "Datum": [Datum, Datum_str,None],
                #    "Time": [["Time"], [r'\b\d{2}:\d{2}:\d{2}\b'],timecheck],
                   "Seite": [Seite,Seite,None]
                   }


def timecheck(txt,main_page_types):
    for i, line in enumerate(txt.splitlines()):
        line = line.decode("utf-8")  # Convert bytes to str
        t_match = re.search(line,main_page_types["Time"][1])
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

def type_check(txt,type2check,main_page_types,line_lim):
    if main_page_types == None:
        match_type_func = main_page_types[type2check][2]
    else:
        # print("type match check auto")
        match_type_func = type_match_check

    for i in range(line_lim):
        # print(i)
        line = txt.splitlines()[i]
        line = line.decode("utf-8")  # Convert bytes to str        
        # first checking if there is the keyword on the line...
        for j in main_page_types[type2check][0]:
        # print(type2check,j)
            t_match_name = match_type_func(line,j)
            if t_match_name:
                # print("Match at %s using %s"%(type2check,j))
                break

        if t_match_name:
            # print("%s found, now looking for more..."%type2check)
            ## check if there is information that matches this info
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

    text = doc[0].get_text()  # This is a string, not bytes
    lines = text.splitlines()
    num_lines = len(lines)
    # print(num_lines)
    len(doc) ## this gives the number of pages
    main_data = ["Datum", "Seite 1"]
    # print(len(doc))
    pages_patch= {}
    main_page = []
    current_main_page = 0
    main_page_b = False
    pages_test = []
    errors=0
    lines2check = 40
    for n,page in enumerate(doc): # iterate the document pages

        main_page = isthismainpage(page,main_page_types,lines2check)

        if any(main_page):
            main_page_b = True

        else:
            main_page_b = False
 
        pages_test.append(main_page_b)
    pages_test=[int(b) for b in pages_test]

    return pages_test


if __name__ == "__main__":
    papath = "data/"
    pdf_files = glob.glob(papath+"*.pdf")
    pdftouse = pdf_files[0]
    # print(pdf_files)
    print('\n\n\n')
    totest = manual_readPDF(pdftouse)

    print("The results using '%s' is: "%pdftouse)
    print(totest)

    # print("The score wrt to Raul's list: {:.2f} %".format(compare_Raul()))

    # for tt in pdf_files:
    #     totest = manual_readPDF(tt)
    #     print("The results using '%s' is: "%tt)
    #     print(totest)