# %%
import os 
import argparse
import subprocess
from bs4 import BeautifulSoup
from modules.Refactor import HTMLProcessor

def main():
    parser = argparse.ArgumentParser(description='Extract <abstract> content from HTML files in a folder.')
    parser.add_argument('folder_path', help='Path to the folder containing HTML files')
    args = parser.parse_args()
    folder_path = args.folder_path
    HTMLProcessor.extract_abstract_from_html_files(folder_path)
    HTMLProcessor.append_abstracts_to_csv(folder_path)

if __name__ == "__main__":
    main()
