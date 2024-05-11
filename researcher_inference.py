import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained("finetuned_PMBERT")
tokenizer = BertTokenizer.from_pretrained("finetuned_PMBERT")

# Read text data from a file
file_path = "new_data.txt"
with open(file_path, "r", encoding="utf-8") as file:
    new_text = file.readlines()

# Tokenize new data
inputs = tokenizer(new_text, padding=True, truncation=True, max_length=128, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

# Convert predictions to labels (assuming binary classification)
predicted_labels = ["Positive" if pred.item() == 1 else "Negative" for pred in predictions]
print(predicted_labels)

