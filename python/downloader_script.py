import argparse
import subprocess
from modules.Downloader import FileManager
from modules.Downloader import NCBIManager
from modules.Downloader import ConfigManager

def main():
    parser = argparse.ArgumentParser(description='Downloads articles, Extracts <body> or <abstract> content')
    parser.add_argument('--body_source', help='Path to the folder containing HTML files')
    parser.add_argument('--is_first_run', action='store_true', help='Flag indicating if this is the first run.')
    parser.add_argument('--threshold', type=int, default=7, help='Time threshold to search for articles in')
    parser.add_argument('--abstract_source', help='Path to the folder containing abstract HTML files')

    args = parser.parse_args()

    body_source = args.body_source
    first_run = args.is_first_run
    threshold = args.threshold
    abstract_source = args.abstract_source

    # Initialize ConfigManager and load configuration
    config_manager = ConfigManager()
    config = config_manager.config

    # Initialize FileManager and NCBIManager with configuration settings
    file_manager = FileManager(data_directory=config.get('data_directory'))
    ncbi_manager = NCBIManager(email=config.get('email'))

    # If body_source is provided, run the Extract_html_body.py script
    if body_source:
        try:
            subprocess.run(['python', './Extract_html_body.py', body_source], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running Extract_html_body.py: {e}")
    # If abstract_source is provided, run the Extract_abstract.py script
    elif abstract_source:
        try:
            subprocess.run(['python', './Extract_abstract.py', abstract_source], check=True)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running Extract_abstract.py: {e}")
    # If neither body_source nor abstract_source is provided, handle date and threshold logic
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
            
            FileManager.overwrite_date(
                date_file_path=date_file_path,
                previous_date=previous_date,
                current_date=current_date,
                is_first_run=first_run, 
                threshold=threshold,
                file_path=config.get('query_file_path'),
                download_dir=config.get('results_directory')
            )

        except Exception as e:
            print(f"An error occurred while handling dates and threshold: {e}")

if __name__ == "__main__":
    main()
