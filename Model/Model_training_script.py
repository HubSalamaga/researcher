import argparse
import os
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel, AdamW
from modules.Classifier import DataPreparator
from modules.Classifier import BertForRegression
from modules.Classifier import ModelTrainer
import sys
import pandas as pd

def main():
        # Set the path for the output file
    cwd = os.getcwd()
    output_file_path = os.path.join(cwd, r"data\initial_training_data\training_output.txt")

    # Redirect standard output to the file
    with open(output_file_path, 'w') as f:
        sys.stdout = f
        torch.cuda.empty_cache()
        cwd = os.getcwd()
        file_path = os.path.join(cwd, r"data\initial_training_data\test.csv")
        dataset = DataPreparator.load_data_from_csv(file_path)

        # Create dataframe before randomisation
        df_before_randomisation = dataset.copy()

        assert not dataset.isnull().values.any(), "Dataset contains NaN values"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scores_to_randomize = dataset[["AwT_score", "SoE_score"]].values
        randomized_scores = DataPreparator.randomize_scores(scores_to_randomize)
        dataset[["AwT_score", "SoE_score"]] = randomized_scores

        # Merge the original and randomized scores into a single dataframe
        df_combined = df_before_randomisation.copy()
        df_combined["randomized_AwT_score"] = randomized_scores[:, 0]
        df_combined["randomized_SoE_score"] = randomized_scores[:, 1]
        pd.set_option('display.max_columns', None)
        print(f"Combined dataframe:\n{df_combined}")

        train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
        
        train_dataloader = DataPreparator.prepare_dataloader(train_data, batch_size=6)
        test_dataloader = DataPreparator.prepare_dataloader(test_data, batch_size=6, test=True)

        num_models = 2
        models = []
        for i in range(num_models):
            model = ModelTrainer.train_model(train_dataloader, device, model_index=i)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            models.append(model)

        avg_predictions, all_predictions, true_scores = ModelTrainer.evaluate_models(models, test_dataloader, device)
        true_scores = pd.DataFrame(true_scores, columns=["AwT_score","SoE_score"])

        # Create the final dataframe for comparison
        df_test_true = test_data.copy()
        df_test_true.rename(columns={"AwT_score": "randomized_AwT_score", "SoE_score": "randomized_SoE_score"}, inplace=True)
        df_avg_predictions = pd.DataFrame(avg_predictions, columns=["predicted_AwT_score", "predicted_SoE_score"])
        df_final_comparison = pd.concat([df_test_true.reset_index(drop=True), df_avg_predictions.reset_index(drop=True)], axis=1)
        # Set display options for pandas
        pd.set_option('display.max_rows', None)
        print(f"Final comparison dataframe:\n{df_final_comparison}")

        true_scores = dataset.drop(test_data.index)
        train_scores_true = dataset.drop(true_scores.index)
        train_scores_true = train_scores_true.iloc[:, 1:3]
        
        print(f"avg predictions: \n {avg_predictions}")

        print(f"Shape of avg_predictions: {avg_predictions.shape}")
        print(f"Shape of true_scores: {train_scores_true.shape}")

        mse = ((avg_predictions - train_scores_true) ** 2).mean(axis=0)
        print(f"Average MSE: {mse}")

if __name__ == "__main__":
    main()

sys.stdout = sys.__stdout__