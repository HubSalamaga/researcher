import pandas as pd

# File paths
input_file_path = r'C:\Users\kacpe\OneDrive\Dokumenty\GitHub\researcher\Model\data\initial_training_data\ocena artykułów do drugiego treningu.csv'  # Path to the input CSV file
output_file_path = r'C:\Users\kacpe\OneDrive\Dokumenty\GitHub\researcher\Model\data\initial_training_data\test.csv'  # Path to the output CSV file


# Read the input CSV file
print(f"Reading input file: {input_file_path}")
input_df = pd.read_csv(input_file_path)
print("Input DataFrame:")
print(input_df.head())

# Drop the 'ID' column
if 'ID' in input_df.columns:
    input_df = input_df.drop(columns=['ID'])
else:
    print("Warning: 'ID' column not found in input file.")

print("Input DataFrame after dropping 'ID' column:")
print(input_df.head())

# Read the output CSV file (if it exists), otherwise create an empty DataFrame with the correct columns
try:
    print(f"Reading output file: {output_file_path}")
    output_df = pd.read_csv(output_file_path)
    print("Output DataFrame before appending:")
    print(output_df.head())
except FileNotFoundError:
    print(f"Output file not found. Creating a new DataFrame with columns: ['text', 'AwT_score', 'SoE_score']")
    output_df = pd.DataFrame(columns=['text', 'AwT_score', 'SoE_score'])

# Append the input data to the output data using pd.concat
output_df = pd.concat([output_df, input_df], ignore_index=True)
print("Output DataFrame after appending:")
print(output_df.head())

# Save the updated output DataFrame back to the CSV file
output_df.to_csv(output_file_path, index=False)
print(f"Data from {input_file_path} has been appended to {output_file_path}.")
