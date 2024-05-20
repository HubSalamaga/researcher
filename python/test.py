# %%
from modules.QueryConst import Const
import argparse
import subprocess
from Bio import Entrez

def main():
    parser = argparse.ArgumentParser(description='Extract <body> content from HTML files in a folder.')
    parser.add_argument('folder_path', help='Path to the folder containing HTML files', nargs='?')
    args = parser.parse_args()
    folder_path = args.folder_path
    if folder_path:
        subprocess.run(['python', './Extract_html_body.py', folder_path])
    else:
        date_file_path, current_date = Const.write_current_date()
        previous_date = Const.read_date(date_file_path=date_file_path)
        Const.overwrite_date(date_file_path=date_file_path,previous_date=previous_date,current_date=current_date)

if __name__ == "__main__":
    main()

#DODAĆ MESH TERMS ŻEBY LEPIEJ TWORZYĆ QUERY (RĘCZNIE!)
#WYCIĄGNĄĆ CYTOWANIA Z ARTYKUŁÓW MOŻE INNE STATYSTYKI DLA AI
#WYCIĄGNĄĆ TYTUŁ ZMIENIĆ NAZWĘ ARTYKUŁU Z PMC_ID
#Ogarnąć konstrukcje folderów tak aby zgadzała się z tym co mamy w modułach i funkcjach 


# %%
