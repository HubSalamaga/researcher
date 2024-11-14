# %%
import argparse
from modules.Downloader import HTMLProcessor

def main():
    parser = argparse.ArgumentParser(description='Extract <abstract> content from HTML files in a folder.')
    parser.add_argument('folder_path', help='Path to the folder containing HTML files')
    args = parser.parse_args()
    folder_path = args.folder_path
    HTMLProcessor.extract_abstract_from_html_files(folder_path)
    HTMLProcessor.append_abstracts_to_csv(folder_path)
    HTMLProcessor.make_abstracts_unique()

if __name__ == "__main__":
    main()
