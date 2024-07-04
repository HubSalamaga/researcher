import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import os
import csv

# Load the CSV file
def load_data_from_csv(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe

# Define paths
cwd = os.getcwd()
file_path = os.path.join(cwd, r'data\inference_data\abstract_dataset.csv')
dataset = load_data_from_csv(file_path)

# Define the BERT regression model
class BertForRegression(nn.Module):
    def __init__(self, model_name, hidden_size=768):
        super(BertForRegression, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.regressor = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_embeddings=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if return_embeddings:
            return pooled_output
        return self.regressor(pooled_output)

# Function to prepare DataLoader
def prepare_dataloader_inference(data, batch_size=6, test=False):
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    inputs = tokenizer(data["Abstract"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not test)
    return dataloader

# Function to load a single model
def load_model(model_path, device):
    model = BertForRegression("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to load all ensemble models
def load_models(model_paths, device):
    models = [load_model(path, device) for path in model_paths]
    return models

# Function to predict using ensemble models
def predict(models, dataloader, device):
    all_predictions = []
    for model in models:
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids, b_input_mask = [item.to(device) for item in batch]
                outputs = model(b_input_ids, b_input_mask)
                predictions.extend(outputs.cpu().numpy())
        all_predictions.append(predictions)
    avg_predictions = np.mean(all_predictions, axis=0)
    return avg_predictions

def filter_and_save_ids(input_file_path, output_file_path):
    with open(output_file_path, mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        if not os.path.isfile(output_file_path):
            writer.writerow(['ID', 'Predicted_AwT_score', 'Predicted_SoE_score'])

        predicted_data = pd.read_csv(input_file_path)
        filtered_data = predicted_data[(predicted_data['Predicted_AwT_score'] >= 0.7) & 
                                       (predicted_data['Predicted_AwT_score'] * predicted_data['Predicted_SoE_score'] >= 0.4)]
        
        filtered_data[['ID', 'Predicted_AwT_score', 'Predicted_SoE_score']].to_csv(output_file_path, index=False)
        print(f"Filtered IDs saved to {output_file_path}")

# Load the ensemble models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_paths = ['trained_model_0.pt', 'trained_model_1.pt']  # Replace with actual paths to your trained models
models = load_models(model_paths, device)

# Prepare dataloader for the new data
batch_size = 15
dataloader = prepare_dataloader_inference(dataset, batch_size=batch_size, test=True)

# Predict using the ensemble models
predictions = predict(models, dataloader, device)

# Add predictions to the dataset
predictions_df = pd.DataFrame(predictions, columns=["Predicted_AwT_score", "Predicted_SoE_score"])
dataset[["Predicted_AwT_score", "Predicted_SoE_score"]] = predictions_df

# Save the predictions to a new CSV file
output_file_path = os.path.join(cwd, r"data/inference_data/predicted_inference.csv")
dataset.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")

# Filter the articles
input_file_path = output_file_path
output_file_path = os.path.join(cwd, r"data/inference_data/filtered_articles.csv")
filter_and_save_ids(input_file_path, output_file_path)
