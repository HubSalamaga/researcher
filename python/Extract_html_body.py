# %%
import os 
import argparse
import subprocess
from bs4 import BeautifulSoup
from modules.Downloader import HTMLProcessor

def main():
    parser = argparse.ArgumentParser(description='Extract <body> content from HTML files in a folder.')
    parser.add_argument('body_source', help='Path to the folder containing HTML files')
    args = parser.parse_args()
    body_source = args.body_source
    HTMLProcessor.extract_body_from_html_files(body_source)

if __name__ == "__main__":
    main()

# %%
