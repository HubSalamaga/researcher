# %%
import os 
import argparse
import subprocess
from bs4 import BeautifulSoup
from modules.QueryConst import Const

def main():
    parser = argparse.ArgumentParser(description='Extract <abstract> content from HTML files in a folder.')
    parser.add_argument('folder_path', help='Path to the folder containing HTML files')
    args = parser.parse_args()
    folder_path = args.folder_path
    Const.extract_abstract_from_html_files(folder_path)

if __name__ == "__main__":
    main()
