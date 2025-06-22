import json
import dateparser
import re
import fitz


def get_dates(text: str):
    pattern = r"""(?x)
    (?<!\d)
    (?:
        # 31-day months
        (?:
        (?:0?[1-9]|[12]\d)|3[01]
        )\s?[./:-][\s.]?
        (?:0?[13578]|1[02]
        |J(?:an(?:uar)?|uli?)
        |M(?:ärz?|ai)
        |Aug(?:ust)?
        |Okt(?:ober)?
        |Dez(?:ember)?
        )\s?(?:[./:-][\s.]?)?
        [1-9]\d\d\d

        | # 30-day months
        (?:
        (?:0?[1-9]|[12]\d)|30
        )\s?[./:-][\s.]?
        (?:0?[13-9]|1[012]
        |J(?:an(?:uar)?|u[nl]i?)
        |M(?:ärz?|ai)
        |A(?:pr(?:il)?|ug(?:ust)?)
        |Sep(?:tember)?
        |Okt(?:ober)?
        |(?:Nov|Dez)(?:ember)?
        )\s?(?:[./:-][\s.]?)?
        [1-9]\d\d\d

        | # February 29 on leap years
        (?:0?[1-9]|[12]\d)\s?[./:-][\s.]?
        (?:0?2|Fe(?:b(?:ruar)?)?)\s?(?:[./:-][\s.]?)?
        [1-9]\d
        (?:[02468][048]|[13579][26])

        | # February other days
        (?:0?[1-9]|[12][0-8])\s?[./:-][\s.]?
        (?:0?2|Fe(?:b(?:ruar)?)?)\s?(?:[./:-][\s.]?)?
        [1-9]\d\d\d
    )
    (?!\d)                        # no digit after
    """
    date_regex = re.compile(pattern, re.VERBOSE | re.IGNORECASE)
    matches = date_regex.findall(text)
    dates = []
    for d in matches:
        if not d:
            continue
        parsed = dateparser.parse(d, languages=['de'])
        if parsed:
            dates.append(parsed.strftime('%Y-%m-%d'))
    return dates



def matching(pdf_file_path: str, vendor_json_path: str):
    pdf = fitz.open(pdf_file_path)
    sap_matches = []
    certainty_scores = []

    # load vendor data
    purchase_orders = {}
    delivery_notes = {}
    dates = {}
    vendors = {}
    street_numbers = {}
    street_names = {}
    zip_codes = {}
    cities = {}

    with open(vendor_json_path, "r") as file:
        vendor_data = json.load(file)
        for i, vendor in enumerate(vendor_data):
            purchase_orders[i] = str(vendor['Purchase Order Number'])
            delivery_notes[i] = str(vendor['Delivery Note Number'])
            raw_date = vendor['Delivery Note Date']
            clean_date = dateparser.parse(raw_date).strftime('%Y-%m-%d')
            dates[i] = clean_date
            vendors[i] = str(vendor['Vendor - Name 1'])
            street_numbers[i] = str(vendor['Vendor - Address - Number'])
            street_names[i] = vendor['Vendor - Address - Street']
            zip_codes[i] = str(vendor['Vendor - Address - ZIP Code'])
            cities[i] = vendor['Vendor - Address - City']
    
    for k in range(len(pdf)):
        page = pdf.load_page(k)
        text = page.get_text()

        match_score = {idx: 0.0 for idx in range(len(vendor_data))}

        generic_pattern = re.compile(r'\b[A-Za-z0-9]+(?:[ _\/-][A-Za-z0-9]+)*\b')
        matches = generic_pattern.findall(text)
        filtered_matches = []
        for match in matches:
            if len(match) < 5:
                continue
            if len(match.split()) > 2:
                continue
            match = match.lower().replace(" ", "").replace("-", "").replace("/", "").strip()
            if any(char.isdigit() for char in match):
                filtered_matches.append(match)
            elif any(char.isupper() for char in match[1:]):
                filtered_matches.append(match)
        for i, purchase_order in purchase_orders.items():
            purchase_order = purchase_order.lower().replace(" ", "").replace("-", "").replace("/", "").strip()
            for match in filtered_matches:
                if match == purchase_order:
                    match_score[i] += 3.5
                    break
    
        for i, delivery_note in delivery_notes.items():
            delivery_note = delivery_note.lower().replace(" ", "").replace("-", "").replace("/", "")
            for match in filtered_matches:
                if match == delivery_note:
                    match_score[i] += 3
                    break

        # match dates
        dates_on_page = get_dates(text)
        for i, vendor_date in dates.items():
            for date in dates_on_page:
                if date == vendor_date:
                    match_score[i] += 2.5
                    break

        
        postal_pattern = re.compile(r'(?<!\d)\d{5}(?!\d)')
        postal_matches = postal_pattern.findall(text)
        for i, vendor_zip_code in zip_codes.items():
            for postal_match in postal_matches:
                if postal_match == vendor_zip_code:
                    match_score[i] += 1
                    break

        for i, city in cities.items():
            if city.lower() in text.lower():
                match_score[i] += 0.5
            
        for i, street_number in street_numbers.items():
            if street_number.lower() in text.lower():
                match_score[i] += 0.25
            
        for i, street_name in street_names.items():
            if street_name is None:
                continue
            if street_name.lower() in text.lower():
                match_score[i] += 0.75

        text_lower = text.lower()
        for vendor_id, vendor_name in vendors.items():
            words = vendor_name.split()
            if not words:
                continue
            # Count how many words appear in the text
            match_count = sum(1 for w in words if w.lower() in text_lower)
            # If at least 75% of the words match, bump the score
            if match_count / len(words) >= 0.75:
                match_score[vendor_id] += 0.75
        max_score = max(match_score.values())
        max_score_vendor = max(match_score, key=match_score.get)
        sap_matches.append(max_score_vendor)
        certainty_scores.append(max_score)
        print(f"Page {k + 1}: Vendor {vendor_data[max_score_vendor]['Vendor - Name 1']}, date {vendor_data[max_score_vendor]['Delivery Note Date']}")

    return sap_matches, certainty_scores
