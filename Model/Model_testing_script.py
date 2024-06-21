import os
import torch
import pandas as pd
from transformers import BertTokenizer
from modules.Classifier import DataPreparator
from modules.Classifier import BertForRegression
from modules.Classifier import ModelLoader
from modules.Classifier import ArticleClassifier

def main():
    cwd = os.getcwd()
    file_path = os.path.join(cwd, r'data\initial_training_data\abstract_dataset.csv')
    dataset = DataPreparator.load_data_from_csv(file_path)

    # Load the ensemble models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_paths = ['trained_model_0.pt', 'trained_model_1.pt']  # Replace with actual paths to your trained models
    models = ModelLoader.load_models(model_paths, device)

    # Prepare dataloader for the new data
    batch_size = 15
    dataloader = DataPreparator.prepare_dataloader_inference(dataset, batch_size=batch_size, test=True)

    # Predict using the ensemble models
    predictions = ArticleClassifier.predict(models, dataloader, device)

    # Add predictions to the dataset
    predictions_df = pd.DataFrame(predictions, columns=["Predicted_AwT_score", "Predicted_SoE_score"])
    dataset[["Predicted_AwT_score", "Predicted_SoE_score"]] = predictions_df

    # Save the predictions to a new CSV file
    output_file_path = os.path.join(cwd, r"data/initial_training_data/predicted_test.csv")
    dataset.to_csv(output_file_path, index=False)

    print(f"Predictions saved to {output_file_path}")

    # Filter the articles
    input_file_path = output_file_path
    output_file_path = os.path.join(cwd, r"data/initial_training_data/filtered_articles.csv")
    ArticleClassifier.filter_and_save_ids(input_file_path, output_file_path)

if __name__ == "__main__":
    main()