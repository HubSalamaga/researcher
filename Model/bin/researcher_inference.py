# %%
import os
import re
import torch
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from transformers import BertTokenizer, BertModel
from nltk.tokenize import sent_tokenize
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn


# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# Define the regression model class
class BertForRegression(nn.Module):
    def __init__(self, model_name, hidden_size=768):
        super(BertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_embeddings=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if return_embeddings:
            return pooled_output  # Return embeddings directly
        return self.regressor(pooled_output)

def clean_text(text):
    # Remove everything between angle brackets
    clean_text = re.sub(r'<[^>]*>', '', text)
    # Remove newline and other extra whitespace characters
    clean_text = clean_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
    # Replace multiple spaces with a single space
    clean_text = re.sub(r'\s+', ' ', clean_text)
    return clean_text

def get_embeddings(text, model, tokenizer, device):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Use the model to get embeddings
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask, return_embeddings=True)
    
    return embeddings.cpu().detach().numpy()  # Convert PyTorch tensor to NumPy array for further processing

def extractive_summarization(text, model, tokenizer, device, num_sentences=5):
    sentences = sent_tokenize(text)
    sentence_embeddings = np.vstack([get_embeddings(sent, model, tokenizer, device)[0].mean(axis=0) for sent in sentences])
    # Clustering sentences
    num_clusters = min(num_sentences, len(sentences))  # Ensuring we don't exceed the number of sentences
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(sentence_embeddings)
    centroids = kmeans.cluster_centers_

    # Selecting one sentence per cluster (closest to centroid)
    summarized_sentences = []
    for centroid in centroids:
        similarities = cosine_similarity([centroid], sentence_embeddings)
        best_sentence = np.argmax(similarities)
        summarized_sentences.append(sentences[best_sentence])

    return " ".join(summarized_sentences)

def process_file(file_path, model, tokenizer, device):
    with open(file_path, "r", encoding='utf-8') as file:
        text = file.read()
    text = clean_text(text)
    summary = extractive_summarization(text, model, tokenizer, device, num_sentences=5)
    return summary

def extract_summaries_from_articles_in_dir(directory_path, model, tokenizer, device):
    summaries = {}
    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(process_file, os.path.join(directory_path, filename), model, tokenizer, device): filename
            for filename in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, filename))
        }
        for future in futures:
            filename = futures[future]
            summaries[filename] = future.result()
    
    return summaries

# Initialize the model with the same model name used during training
model = BertForRegression("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model.load_state_dict(torch.load(r"C:\Users\Hubert\Documents\GitHub\researcher\Model\finetuned_PMBERT_regression.pth"))

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Directory containing text files
directory_path = r"C:\Users\Hubert\Documents\GitHub\researcher\Model\data\test_articles"

# Process all files in the directory and get summaries
summaries = extract_summaries_from_articles_in_dir(directory_path, model, tokenizer, device)

# Print summaries
for filename, summary in summaries.items():
    print(f"Summary of file {filename}:\n{summary}\n")



# %%




