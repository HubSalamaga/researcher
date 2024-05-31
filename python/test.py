import argparse
import subprocess
from modules.QueryConst import Const  # Assuming Const is a module you have defined
from Bio import Entrez

def main():
    parser = argparse.ArgumentParser(description='Extract <body> content from HTML files in a folder.')
    parser.add_argument('--folder_path', help='Path to the folder containing HTML files')
    parser.add_argument('--is_first_run', action='store_true', help='Flag indicating if this is the first run.')
    parser.add_argument('--threshold', type=int, default=7, help='Time threshold to search for articles in')
    parser.add_argument('--abstract', help='Path to the folder containing abstract HTML files')

    args = parser.parse_args()

    folder_path = args.folder_path
    first_run = args.is_first_run
    threshold = args.threshold
    abstract_source = args.abstract

    # If folder_path is provided, run the Extract_html_body.py script
    if folder_path:
        try:
            subprocess.run(['python', './Extract_html_body.py', folder_path], check=True)
            print(f"Folder path provided {folder_path}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running Extract_html_body.py: {e}")
    # If abstract_source is provided, run the Extract_abstract_body.py script
    elif abstract_source:
        try:
            subprocess.run(['python', './Extract_abstract_body.py', abstract_source], check=True)
            print(f"Abstract path is {abstract_source}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running Extract_abstract_body.py: {e}")
    # If neither folder_path nor abstract_source is provided, handle date and threshold logic
    else:
        try:
            date_file_path, current_date = Const.write_current_date()
            previous_date = Const.read_date(date_file_path=date_file_path)
            Const.overwrite_date(
                date_file_path=date_file_path,
                previous_date=previous_date,
                current_date=current_date,
                is_first_run=first_run,
                threshold=threshold
            )
        except Exception as e:
            print(f"An error occurred while handling dates and threshold: {e}")

if __name__ == "__main__":
    main()

# TODO:
# Add MeSH terms to better create queries (manually!)
# Extract citations from articles and possibly other statistics for AI
# Extract the title and change the article name from PMC_ID
