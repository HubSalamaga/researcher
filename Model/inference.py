import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
import os

# Load the CSV file
def load_data_from_csv(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe

# Define paths
cwd = os.getcwd()
file_path = os.path.join(cwd, r"data/initial_training_data/test.csv")
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
def prepare_dataloader(data, batch_size=6, test=False):
    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    inputs = tokenizer(data["text"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not test)
    return dataloader

# Function to load the model
def load_model(model_path, device):
    model = BertForRegression("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Function to predict using the model
def predict(model, dataloader, device):
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            b_input_ids, b_input_mask = [item.to(device) for item in batch]
            outputs = model(b_input_ids, b_input_mask)
            predictions.extend(outputs.cpu().numpy())
    return predictions

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'path_to_your_trained_model.pt'  # Replace with the actual path to your trained model
model = load_model(model_path, device)

# Prepare dataloader for the new data
batch_size = 15
dataloader = prepare_dataloader(dataset, batch_size=batch_size, test=True)

# Predict
predictions = predict(model, dataloader, device)

# Add predictions to the dataset
predictions_df = pd.DataFrame(predictions, columns=["AwT score", "SoE score"])
dataset[["Predicted AwT score", "Predicted SoE score"]] = predictions_df

# Save the predictions to a new CSV file
output_file_path = os.path.join(cwd, r"data/initial_training_data/predicted_test.csv")
dataset.to_csv(output_file_path, index=False)

print(f"Predictions saved to {output_file_path}")
