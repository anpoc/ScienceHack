from text_based_split.predict import predict
from ner_second_stage.ner import data_processor
import argparse

def main(pdf_path, sap_data_path, output_file_name):
    list_start_new_files = predict(pdf_path)

    data_processor(sap_data_path, pdf_path, list_start_new_files, output_file_name)

if __name__ == "__main__":
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
    out_file_name = args.pdf_path.split('/')[-1].split('.')[0] + ".json"
    main(args.pdf_path, args.json_path, out_file_name)