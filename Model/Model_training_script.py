import argparse
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW
from modules.Classifier import DataPreparator
from modules.Classifier import BertForRegression
from modules.Classifier import ModelTrainer

def main():
    # parser = argparse.ArgumentParser(description='Trains model on training dataset containing abstracts, AwT and SoE scores as a csv file')
    # parser.add_argument('file_path', help='Path to the folder containing training dataset')
    # args = parser.parse_args()
    # file_path = args.file_path


    cwd = os.getcwd()
    file_path = os.path.join(cwd, r"data\initial_training_data\test.csv")
    dataset = DataPreparator.load_data_from_csv(file_path)

    train_y = dataset[['AwT_score', 'SoE_score']].values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores_to_randomize = dataset[["AwT_score", "SoE_score"]].values
    randomized_scores = DataPreparator.randomize_scores(scores_to_randomize)
    dataset[["AwT_score", "SoE_score"]] = randomized_scores
    train_data, test_data = train_test_split(dataset, test_size=0.4, random_state=42)

    tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    train_dataloader = DataPreparator.prepare_dataloader(train_data,batch_size=15)
    test_dataloader = DataPreparator.prepare_dataloader(test_data, batch_size=15, test=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_models = 2
    models = [ModelTrainer.train_model(train_dataloader, device, model_index=i) for i in range(num_models)]

    avg_predictions, all_predictions, true_scores = ModelTrainer.evaluate_models(models, test_dataloader, device)

    # Assuming true_scores is already defined correctly
    true_scores = dataset.drop(test_data.index)
    train_scores_true = dataset.drop(true_scores.index)
    train_scores_true = train_scores_true.iloc[:,1:3]
    print(train_scores_true)
    #Check the shape of avg_predictions
    print(f"Shape of avg_predictions: {avg_predictions.shape}")
    print(f"Shape of true_scores: {train_scores_true.shape}")

    mse = ((avg_predictions - train_scores_true) ** 2).mean(axis=0)
    print(f"Average MSE: {mse}")


if __name__ == "__main__":
    main()


