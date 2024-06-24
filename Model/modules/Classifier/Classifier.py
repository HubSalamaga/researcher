import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import expon, loguniform, uniform
import os
import torch.nn.functional as F
import csv

class DataPreparator:
    def load_data_from_csv(file_path):
        dataframe = pd.read_csv(file_path)
        return dataframe

    def randomize_scores(scores, max_deviation= 0.05):
        randomized_scores = scores * (1 + np.random.uniform(-max_deviation,max_deviation,size=scores.shape))
        return np.clip(randomized_scores,0,1)
    
    def prepare_dataloader(data, batch_size=6, test=False):
        tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        inputs = tokenizer(data["text"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
        labels = torch.tensor(data[["AwT_score", "SoE_score"]].values).float()
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not test)
        return dataloader
    
    def prepare_dataloader_inference(data, batch_size=6, test=False):
        tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        inputs = tokenizer(data["Abstract"].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")
        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not test)
        return dataloader

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
    
class ModelTrainer:
    def save_model(model, path):
        torch.save(model.state_dict(), path)

    def train_model(train_dataloader, device, epochs = 16, model_index = 0):
        model = BertForRegression("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5) # test value  # torch.optim.AdamW
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_dataloader:
                b_input_ids, b_input_mask, b_labels = [item.to(device) for item in batch] #co to jest b_input_ids
                optimizer.zero_grad()
                outputs = model(b_input_ids,b_input_mask) #co to jest????
                #loss = criterion(outputs.squeeze(),b_labels)
                loss = criterion(outputs,b_labels)  # Use the custom loss function #outputs = predictions , b_labels = targets 
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}, Loss: {avg_train_loss}")
            ModelTrainer.save_model(model, f"trained_model_{model_index}.pt")
        
        return model

    def evaluate_models(models, test_dataloader, device):
        all_predictions = []
        true_scores = []
        for model in models:
            model.eval()
            predictions = []
            with torch.no_grad():
                for batch in test_dataloader:
                    b_input_ids, b_input_mask, b_labels = [item.to(device) for item in batch]
                    outputs = model(b_input_ids, b_input_mask)
                    predictions.extend(outputs.cpu().numpy())
                    true_scores.extend(b_labels.cpu().numpy())
            all_predictions.append(predictions)

        avg_predictions = np.mean(all_predictions, axis=0)
        true_scores = np.array(true_scores)
        return avg_predictions, all_predictions, true_scores
    
class ModelLoader:
    def load_model(model_path, device):
        model = BertForRegression("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model

    # Function to load all ensemble models
    def load_models(model_paths, device):
        models = [ModelLoader.load_model(path, device) for path in model_paths]
        return models
    
class ArticleClassifier:
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


    