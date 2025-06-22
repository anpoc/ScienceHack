import argparse
import json
import os

from matching import matching
from predict_splits import predict as predict_split


def main():
    """Main function that processes PDF and JSON files."""
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

    split_pred = predict_split(args.pdf_path)
    matching_results, certainty_scores = matching(args.pdf_path, args.json_path)

    with open(args.json_path, "r") as file:
        vendor_data = json.load(file)

    confidence = "high" if certainty_scores[0] >= 7 else "medium" if certainty_scores[0] >= 4.0 else "low"
    records = [
        {
            "page": 0,
            "confidence": confidence,
            "MBLNR": vendor_data[matching_results[0]]['MBLNR'],
            "MJAHR": vendor_data[matching_results[0]]['MJAHR'],
        }
    ]
    current_vendor = matching_results[0]
    for i in range(len(split_pred)):
        if certainty_scores[i] >= 4.0 and matching_results[i] != current_vendor:
            confidence = "high" if certainty_scores[i] >= 7 else "medium"
            records.append({
                "page": i,
                "confidence": confidence,
                "MBLNR": vendor_data[matching_results[i]]['MBLNR'],
                "MJAHR": vendor_data[matching_results[i]]['MJAHR'],
            })
            current_vendor = matching_results[i]
            continue
        if certainty_scores[i] >= 4.0 and matching_results[i] == current_vendor:
            continue
        if split_pred[i] == 1:
            records.append({
                "page": i,
                "confidence": "low",
                "MBLNR": vendor_data[matching_results[i]]['MBLNR'],
                "MJAHR": vendor_data[matching_results[i]]['MJAHR'],
            })
    
    os.makedirs(f"./results/n_method", exist_ok=True)
    with open(f"./results/n_method/{args.pdf_path.split('/')[-1].split('.')[0]}.json", "w") as file:
        json.dump(records, file, indent=4)


if __name__ == "__main__":
    main() 