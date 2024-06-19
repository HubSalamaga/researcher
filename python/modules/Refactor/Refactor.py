import os
import datetime

class FileManager:
    def __init__(self, data_directory):
        self.data_directory = data_directory

    def write_current_date(self):
        current_date = datetime.datetime.now().date()
        cwd = os.getcwd()
        data_path = os.path.join(cwd, self.data_directory)
        dir_path_date = os.path.join(data_path, "date")
        date_file_path = os.path.join(dir_path_date, "current_date.txt")
        
        if not os.path.exists(dir_path_date):
            os.makedirs(dir_path_date)
            with open(date_file_path, "w") as date_file:
                date_file.write(str(current_date))
        else: 
            return date_file_path, current_date


    @staticmethod
    def read_date(date_file_path):
        if os.path.exists(date_file_path):
            with open(date_file_path, "r") as date_file:
                previous_date = date_file.read()
            return previous_date
        return None

    @staticmethod
    def overwrite_date(date_file_path, previous_date, current_date, file_path, download_dir, is_first_run=True, threshold=7):
        current_date = datetime.datetime.strptime(str(current_date), "%Y-%m-%d").date()
        previous_date = datetime.datetime.strptime(str(previous_date), "%Y-%m-%d").date()
        date_difference = abs((current_date - previous_date).days)
        cwd = os.getcwd()
        results_directory = os.path.join(cwd, "results")
        download_dir = results_directory

        try:
            if is_first_run:
                file_path = os.path.join(cwd, "data", "query_list", "queries.txt")
                NCBIManager.read_queries_and_fetch_articles(file_path, download_dir, current_date, previous_date)
                print(f"First run success")
            elif date_difference == 0:
                print("Program run on the same day")
            elif date_difference >= threshold:
                file_path = os.path.join(cwd, "data", "query_list", "queries.txt")
                NCBIManager.read_queries_and_fetch_articles(date_file_path, file_path, download_dir, current_date, previous_date)
                with open(date_file_path, 'w') as file:
                    file.write(str(current_date))
                print(f"More than {threshold} days passed")
            else:
                print(f"Less than {threshold} days passed")
        except Exception as e:
            print(f"An error has occurred: {e}")

    @staticmethod
    def convert_date_format(date_str, from_format, to_format):
        date_obj = datetime.datetime.strptime(date_str, from_format)
        return date_obj.strftime(to_format)
    
from Bio import Entrez
import time
import os
import hashlib
import re
import datetime

class NCBIManager:
    def __init__(self, email):
        self.email = email
        Entrez.email = email

    @staticmethod
    def sanitize_query(query, max_length=50):
        # Truncate and sanitize the query for folder name
        sanitized_query = re.sub(r'[<>:"/\\|?*]', '', query.replace(' ', '_'))
        truncated_query = sanitized_query[:max_length]
        return truncated_query

    @staticmethod
    def hash_query(query):
        # Create a hash of the query for folder name
        return hashlib.md5(query.encode()).hexdigest()[:8]

    @staticmethod
    def generate_folder_name(query):
        sanitized_query = NCBIManager.sanitize_query(query)
        query_hash = NCBIManager.hash_query(query)
        return f"{sanitized_query}_{query_hash}"

    @staticmethod
    def fetch_pmc_ids(query):
        print(f"Fetching PMC IDs for query: {query}")  # Debug statement
        handle = Entrez.esearch(db="pmc", term=query, retmax=50)
        record = Entrez.read(handle)
        handle.close()
        print(f"Fetched IDs: {record['IdList']}")  # Debug statement
        return record['IdList']

    @staticmethod
    def create_query_folder(query, base_dir):
        folder_name = NCBIManager.generate_folder_name(query)
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return folder_path, True  # Return True indicating it's a new folder
        return folder_path, False  # Return False indicating it's an existing folder

    @staticmethod
    def get_last_modification_date(folder_path):
        try:
            modification_time = os.path.getmtime(folder_path)
            return datetime.datetime.fromtimestamp(modification_time).date()
        except Exception as e:
            print(f"An error occurred while getting last modification date: {e}")
            return None

    @staticmethod
    def read_queries_and_fetch_articles(date_file_path, file_path, download_dir, current_date, previous_date):
        from_format = "%Y-%m-%d"
        to_format = "%Y/%m/%d"

        current_date_str = FileManager.convert_date_format(str(current_date), from_format, to_format)

        with open(file_path, 'r') as file:
            for query in file:
                query = query.strip()
                query_folder, is_new_folder = NCBIManager.create_query_folder(query, download_dir)

                #Check if the folder already exists and get the last modification date
                if not is_new_folder:
                    last_modification_date = NCBIManager.get_last_modification_date(query_folder)
                    if last_modification_date:
                        previous_date = last_modification_date

                previous_date_str = FileManager.convert_date_format(str(previous_date), from_format, to_format)

                if is_new_folder or current_date_str != previous_date_str:
                    constructed_query = f"{query} AND ({previous_date_str}[PubDate] : {current_date_str}[PubDate])"
                    print(f"Constructed query: {constructed_query}")  # Debug statement
                    if constructed_query:
                        print(f"Processing query: {constructed_query}")
                        print(f"Creating/using folder: {query_folder}")
                        NCBIManager.download_pmc_articles(constructed_query, query_folder)

    @staticmethod
    def download_pmc_articles(query, destination_dir, max_retries=3, rate_limit=0.33, timeout=30):
        pmc_ids = NCBIManager.fetch_pmc_ids(query)
        if not pmc_ids:
            print(f"No articles found for query: {query}")
            return

        start_time = time.time()
        for pmc_id in pmc_ids:
            retries = 0
            success = False
            while retries < max_retries and not success:
                try:
                    handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="xml")
                    article_content = handle.read()
                    handle.close()
                    article_path = os.path.join(destination_dir, f"{pmc_id}.xml")
                    with open(article_path, 'wb') as article_file:
                        article_file.write(article_content)
                        print(f"Article {pmc_id} downloaded.")
                    success = True
                    start_time = time.time()  # Reset the timer after a successful download
                except Exception as e:
                    print(f"Error downloading article {pmc_id}: {e}. Retrying...")
                    retries += 1
                    time.sleep(rate_limit)
                finally:
                    if not success and retries >= max_retries:
                        time.sleep(rate_limit)
            
            # Check if timeout has been exceeded
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print(f"Timeout exceeded. Restarting from article {pmc_id}.")
                download_pmc_articles(query, destination_dir, max_retries, rate_limit, timeout)
                return

        print(f"Downloaded {len(pmc_ids)} articles to {destination_dir}")

    @staticmethod
    def sanitize_for_windows_folder_name(query):
        cleaned_query = re.sub(r'\[.*?\]|\(.*?\)', '', query)
        cleaned_query = cleaned_query.replace(' ', '_')
        invalid_chars = r'[<>:"/\\|?*]'
        sanitized_query = re.sub(invalid_chars, '', cleaned_query)
        sanitized_query = re.sub(r'_+', '_', sanitized_query)
        return sanitized_query.strip('_')

from bs4 import BeautifulSoup
import os
import csv

class HTMLProcessor:
    @staticmethod
    def extract_body_from_html_files(folder_path):
        if not os.path.isdir(folder_path):
            raise ValueError("The provided path is not a valid directory")
        
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    body = soup.find('body')
                    if body:
                        body_path = os.path.join(folder_path, "html_bodies")
                        if not os.path.exists(body_path):
                            os.makedirs(body_path)
                        output_file_path = os.path.join(body_path, f"{os.path.splitext(filename)[0]}_body.html")
                        with open(output_file_path, 'w', encoding='utf-8') as output_file:
                            output_file.write(str(body))
                    else:
                        print(f"No body tag found")

    @staticmethod
    def extract_abstract_from_html_files(folder_path):
        if not os.path.isdir(folder_path):
            raise ValueError("The provided path is not a valid directory")
        
        abstract_contents = {}
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        soup = BeautifulSoup(file, 'html.parser')
                        abstract = soup.find('abstract')
                        if abstract:
                            p_tags = abstract.find_all('p')
                            abstract_text = ' '.join(p.get_text() for p in p_tags)
                                
                            abstract_path = os.path.join(folder_path, "txt_abstracts")
                            if not os.path.exists(abstract_path):
                                os.makedirs(abstract_path)
                                print(f"Created directory: {abstract_path}")
                            output_file_path = os.path.join(abstract_path, f"{os.path.splitext(filename)[0]}_abstract.txt")
                            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                                output_file.write(str(abstract_text))
                                print(f"Written abstract to: {output_file_path}")
                            abstract_contents[file_path] = str(abstract)
                        else:
                            print(f"No abstract tag found in file: {file_path}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        return abstract_contents
    
    @staticmethod
    
    def append_abstracts_to_csv(folder_path):
        txt_folder = os.path.join(folder_path, "txt_abstracts")
        cwd = os.getcwd()
        parent_dir = os.path.dirname(cwd)
        abstracts_dataset_path = os.path.join(parent_dir, r"Model\data\initial_training_data")
        # Ensure the CSV folder exists
        if not os.path.exists(abstracts_dataset_path):
            os.makedirs(abstracts_dataset_path)
        
        csv_file_path = os.path.join(abstracts_dataset_path, 'abstract_dataset.csv')
        
        # Check if the CSV file exists, create it if it doesn't
        file_exists = os.path.isfile(csv_file_path)
        
        with open(csv_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            
            # Write header if the file is being created
            if not file_exists:
                writer.writerow(['Abstract', 'ID'])
            
            for filename in os.listdir(txt_folder):
                if filename.endswith('_abstract.txt'):
                    file_id = filename.split('_')[0]
                    file_path = os.path.join(txt_folder, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as file:
                        lines = file.readlines()
                        # Ignore empty lines at the beginning
                        abstract_lines = [line.strip() for line in lines if line.strip()]
                        abstract_text = ' '.join(abstract_lines)
                        
                        if abstract_text:  # Ensure we don't write empty abstracts
                            writer.writerow([abstract_text, file_id])
                            print(f"Appended abstract from file {filename} to CSV.")

import json
import os

class ConfigManager:
    def __init__(self, config_path='config.json'):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as config_file:
                return json.load(config_file)
        else:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

    def get(self, key, default=None):
        return self.config.get(key, default)