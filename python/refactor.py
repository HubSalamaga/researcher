import argparse
import subprocess
from modules.Refactor import FileManager
from modules.Refactor import NCBIManager
from modules.Refactor import HTMLProcessor
from modules.Refactor import ConfigManager

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

    # Initialize ConfigManager and load configuration
    config_manager = ConfigManager()
    config = config_manager.config

    # Initialize FileManager and NCBIManager with configuration settings
    file_manager = FileManager(data_directory=config.get('data_directory'))
    ncbi_manager = NCBIManager(email=config.get('email'))

    # If folder_path is provided, run the Extract_html_body.py script
    if folder_path:
        try:
            subprocess.run(['python', './Extract_html_body.py', folder_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running Extract_html_body.py: {e}")
    # If abstract_source is provided, run the Extract_abstract_body.py script
    elif abstract_source:
        try:
            subprocess.run(['python', './Extract_abstract_body.py', abstract_source], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running Extract_abstract_body.py: {e}")
    # If neither folder_path nor abstract_source is provided, handle date and threshold logic
    else:
        try:
            date_file_path, current_date = file_manager.write_current_date()
            print(f"Date file path: {date_file_path}, Current date: {current_date}")
            previous_date = file_manager.read_date(date_file_path=date_file_path)
            print(f"Previous date: {previous_date}")

            if previous_date is None:
                if first_run:
                    previous_date = current_date
                else:
                    raise ValueError("Previous date not found and not the first run. Please check the date file.")
            
            ncbi_manager.read_queries_and_fetch_articles(
                file_path=config.get('query_file_path'),
                download_dir=config.get('results_directory'),
                current_date=current_date,
                previous_date=previous_date
            )
        except Exception as e:
            print(f"An error occurred while handling dates and threshold: {e}")

if __name__ == "__main__":
    main()
