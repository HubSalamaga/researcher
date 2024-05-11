import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split

# Load annotated dataset
dataset = pd.read_csv("researcher_training.csv")


# Split dataset into train and test sets
train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Load PubMedBERT tokenizer
tokenizer = BertTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

# Tokenize input data
train_inputs = tokenizer(train_data["text"].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
test_inputs = tokenizer(test_data["text"].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")

# Convert labels to tensor
train_labels = torch.tensor(train_data["label"].tolist())
test_labels = torch.tensor(test_data["label"].tolist())

# Define PubMedBERT model for sequence classification
model = BertForSequenceClassification.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", num_labels=2)

# Define optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=2e-5)

# Define training parameters
epochs = 3
batch_size = 16

# Fine-tune PubMedBERT model
for epoch in range(epochs):
    for i in range(0, len(train_inputs["input_ids"]), batch_size):
        optimizer.zero_grad()
        batch_inputs = {k: v[i:i+batch_size] for k, v in train_inputs.items()}
        batch_labels = train_labels[i:i+batch_size]
        outputs = model(**batch_inputs, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the fine-tuned model on the test set
with torch.no_grad():
    outputs = model(**test_inputs)
    predictions = torch.argmax(outputs.logits, dim=1)

# Calculate accuracy
accuracy = (predictions == test_labels).sum().item() / len(test_labels)
print(f"Accuracy: {accuracy}")

# Save the fine-tuned model
model.save_pretrained("finetuned_PMBERT")
