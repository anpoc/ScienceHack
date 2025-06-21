from datetime import datetime
import json
import math
import numpy as np
from pypdf import PdfReader
from transformers import pipeline
import re
import Levenshtein
import scipy

import fitz

def match_with_vendor_name_and_order_number_and_delivery_numnber_regex(pdf_file_path: str, vendor_data_file_path: str):
    # for each chunk text try to regex the vendor name, if we find multiple matches return all of them,
    # then distance is only considered for those

    doc = fitz.open(pdf_file_path)

    # Group text by document chunks
    doc_texts = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        doc_texts.append(page.get_text())

    with open(vendor_data_file_path, "r") as file:
        vendor_data = json.load(file)
        all_vendor_names_with_indices = [(i, vendor["Vendor - Name 1"], vendor['Purchase Order Number'], vendor['Delivery Note Number']) for i, vendor in enumerate(vendor_data)]

    # for each doc_text, try to find vendor names using regex
    possible_vendor_indices = []
    for doc_text in doc_texts:
        found_indices = set()
        for index, vendor_name, _, _ in all_vendor_names_with_indices:
            pattern = re.compile(re.escape(vendor_name), re.IGNORECASE)
            if pattern.search(doc_text):
                found_indices.add(index)
        
        if len(found_indices) == 0: # if no vendor names were found add all indices, since it could be any vendor
            found_indices = set(range(len(vendor_data)))

        possible_vendor_indices.append(list(found_indices))

    # now try to to match the regex with purchase order number
    # for i, doc_text in enumerate(doc_texts):
    #     found_indices = set()
    #     for index, _, purchase_order_number, _ in all_vendor_names_with_indices:
    #         pattern = re.compile(re.escape(str(purchase_order_number)), re.IGNORECASE)
    #         if pattern.search(doc_text):
    #             found_indices.add(index)
    #     if len(found_indices) == 0:  # if no purchase order numbers were found, keep the previous indices
    #         found_indices = set(possible_vendor_indices[i])  # keep the previous indices
    #     else:
    #         # intersect with previous indices to narrow down the possible vendors
    #         found_indices = found_indices.intersection(possible_vendor_indices[i])
    #         if len(found_indices) == 0:  # if no purchase order numbers were found, keep the previous indices
    #             found_indices = set(possible_vendor_indices[i])  # keep the previous indices
    #     possible_vendor_indices[i] = list(found_indices)

    # print possible vendor indices for each chunk
    for i, indices in enumerate(possible_vendor_indices):
        print(f"Page {i + 1}: Possible vendor indices: {indices}")

    return possible_vendor_indices
 

def match_page_to_customers(pdf_file_path: str, vendor_data_file_path: str, llm_extraction_json_path: dict):

    possible_vendor_indices_by_page = match_with_vendor_name_and_order_number_and_delivery_numnber_regex(pdf_file_path, vendor_data_file_path)
    
    page_infos = []
    with open(llm_extraction_json_path, "r") as file:
        llm_extraction_json = json.load(file)
        llm_extraction_data = [llm_extraction_json[key] for key in sorted(llm_extraction_json, key=lambda k: int(k))]
        for page_data in llm_extraction_data:
            order_number = page_data["Purchase Order Number"]
            delivery_number = page_data["Delivery Note Number"]
            date = page_data["Delivery Note Date"]
            if date != 'NA':
                date = datetime.strptime(page_data["Delivery Note Date"], "%Y-%m-%d").strftime("%d.%m.%Y")
            vendor_name = page_data["Vendor - Name 1"]
            vendor_address = page_data['Vendor - Address']

            if isinstance(order_number, list):
                order_number = order_number[0]
            if isinstance(delivery_number, list):
                delivery_number = delivery_number[0]
            if isinstance(date, list):
                date = date[0]
            if isinstance(vendor_name, list):
                vendor_name = vendor_name[0]
            if isinstance(vendor_address, list):
                vendor_address = vendor_address[0]
            page_infos.append({
                "order_number": order_number,
                "delivery_number": delivery_number,
                "date": date,
                "vendor_name": vendor_name,
                "vendor_address": vendor_address
            })
        
    # collect date, vendor name and address from vendor data
    vendor_infos = []
    with open(vendor_data_file_path, "r") as file:
        vendor_data = json.load(file)
        # vendor_data = vendor_data[:2]
        for vendor in vendor_data:
            vendor_infos.append({
                "order_number": vendor["Purchase Order Number"],
                "delivery_number": vendor["Delivery Note Number"],
                "date": datetime.strptime(vendor["Delivery Note Date"], "%Y-%m-%dT%H:%M:%S.%f").strftime("%d.%m.%Y"),
                "name": vendor["Vendor - Name 1"],
                "address": f"{vendor["Vendor - Address - Street"]} {vendor["Vendor - Address - Number"]}\n{vendor["Vendor - Address - ZIP Code"]} {vendor["Vendor - Address - City"]}\n{vendor["Vendor - Address - Country"]}",
            })
            
    # calculate distances between page infos and vendor infos in each "dimension" (date, name, address)
    distance_matrix = []
    for page_info in page_infos:
        distances = []
        for vendor_info in vendor_infos:
            order_number_differs = not (page_info["order_number"] == vendor_info["order_number"])
            delivery_number_differs = not(page_info["delivery_number"] == vendor_info["delivery_number"])
            date_distance_value = date_distance(page_info["date"], vendor_info["date"]) if page_info["date"] and vendor_info["date"] else float('inf')
            name_distance_value = levenshtein_distance(page_info["vendor_name"], vendor_info["name"]) if page_info["vendor_name"] and vendor_info["name"] else float('inf')
            address_distance_value = levenshtein_distance(page_info["vendor_address"], vendor_info["address"]) if page_info["vendor_address"] and vendor_info["address"] else float('inf')
            
            distances.append({
                "order_number_differs": order_number_differs,
                "delivery_number_differs": delivery_number_differs,
                "date_distance": date_distance_value,
                "name_distance": name_distance_value,
                "address_distance": address_distance_value})
        distance_matrix.append(distances)
        
    # normalize distance matrix for each dimension (date, name, address)
    min_date_distance = min(min(d['date_distance'] for d in distances) for distances in distance_matrix)
    max_date_distance = max(max(d['date_distance'] for d in distances) for distances in distance_matrix)
    min_name_distance = min(min(d['name_distance'] for d in distances) for distances in distance_matrix)
    max_name_distance = max(max(d['name_distance'] for d in distances) for distances in distance_matrix)
    min_address_distance = min(min(d['address_distance'] for d in distances) for distances in distance_matrix)
    max_address_distance = max(max(d['address_distance'] for d in distances) for distances in distance_matrix)
    for i, distances in enumerate(distance_matrix):
        for j, distance in enumerate(distances):
            distance['date_distance'] = (distance['date_distance'] - min_date_distance) / (max_date_distance - min_date_distance)
            distance['name_distance'] = (distance['name_distance'] - min_name_distance) / (max_name_distance - min_name_distance)
            distance['address_distance'] = (distance['address_distance'] - min_address_distance) / (max_address_distance - min_address_distance)
          
    # calculate overall distance as a weighted sum of the individual distances
    dm = []
    for distances in distance_matrix:
        overall_distances = []
        for distance in distances:
            # if the order number or delivery number is the same, we assume a perfect match
            # otherwise we calculate the overall distance as the average of the individual distances
            if not distance['order_number_differs'] or not distance['delivery_number_differs']:
                overall_distances.append(0)
                continue

            overall_distance = distance['date_distance'] + distance['name_distance'] + distance['address_distance']
            overall_distance /= 3
            overall_distances.append(overall_distance)
        dm.append(overall_distances)
        
    dm = np.array(dm) # shape (num_pages, num_vendors)
    
    # only consider vendors that were found in the regex step
    # Create a copy to mask disallowed indices
    masked_dm = np.full_like(dm, np.inf, dtype=float)
    for i, allowed in enumerate(possible_vendor_indices_by_page):
        masked_dm[i, allowed] = dm[i, allowed]

    # Now take argmin on the masked matrix
    predictions = np.argmin(masked_dm, axis=1)
    
    # compute confidences by using the inverse of the distance and applying a softmax function
    confidences = scipy.special.softmax(-dm, axis=1)
    predicted_confidences = confidences[np.arange(len(predictions)), predictions]
    
    print("Predictions for each page:")
    for i, (pred, conf) in enumerate(zip(predictions, predicted_confidences)):
        print(f"page {i + 1}: Vendor {pred + 1}, confidence {conf:4f} (Distance: {dm[i][pred]})")
        
    return predictions


def construct_output_from_predictions(chunk_prediction: list, predictions: list, vendor_data) -> list:
    """
    Constructs a JSON output from the predictions and vendor data.
    
    Args:
        chunk_prediction (list): binary indication whether to split e.g. [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
        predictions(list): index of the vendors [5, 4, 2] so first chunk corresponds to 5th vendor
        vendor_data(list of dicts): list of vendors
    """
    output = []
    chunk_start_points = [i for i, val in enumerate(chunk_prediction) if val == 1]
    for i, (start_point, pred) in enumerate(zip(chunk_start_points, predictions)):
        vendor_info = vendor_data[pred]
        output.append({
            "start": start_point,
            "MBLNR": vendor_info['MBLNR'],
            "MJAHR": vendor_info["MJAHR"]
        })
    return output
        
def date_distance(date1: str, date2: str) -> int:

    date_format = "%d.%m.%Y"
    try:
        d1 = datetime.strptime(date1, date_format)
        d2 = datetime.strptime(date2, date_format)
    
        return abs((d1 - d2).days)
    except ValueError:
        # If the date format is incorrect, return a large distance
        return float(365*10)

def levenshtein_distance(str1: str, str2: str) -> int:
    # Use Levenshtein distance to measure the similarity between two strings
    return Levenshtein.distance(str1.lower(), str2.lower())


if __name__ == "__main__":
    # Example usage
    pdf_file_path = "../BECONEX_challenge_materials_samples/batch_1_2017_2018.pdf"
    customer_data_file_path = "../BECONEX_challenge_materials_samples/SAP_data.json"
    llm_extraction_json_path = "src/colpali/cl_results_batch_1_2017_2018.pdf.json"
    
    ground_truth = [6, 4, 4, 4, 4, 2, 2, 7, 7, 7, 7, 0, 1, 10, 3, 9]
    
    prediction = match_page_to_customers(pdf_file_path, customer_data_file_path, llm_extraction_json_path)
    
    
    print("Ground truth:", ground_truth)
    print("prediction:", prediction)
    print("Accuracy:", np.mean(np.array(ground_truth) == np.array(prediction)))
    
    