# using model  distilbert-base-uncased

import pandas as pd
import re
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

data1 = pd.read_csv('Truth_Seeker_Model_Dataset.csv')
data1 = data1.drop(['Unnamed: 0', "target"], axis=1)

columns_to_encode = ['author', 'statement', 'manual_keywords', "5_label_majority_answer", "3_label_majority_answer"]
le = LabelEncoder()
for col in columns_to_encode:
    data1[col] = le.fit_transform(data1[col])

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)    # Remove mentions
    text = re.sub(r'#\w+', '', text)    # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    text = text.lower()                 # Convert to lowercase
    return text

data1['tweet'] = data1['tweet'].apply(clean_text)

tokenizer1 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_text(tokenizer, texts):
    tokenized = texts.apply(lambda x: tokenizer(x, padding='max_length', truncation=True, max_length=128, return_tensors="pt"))
    input_ids = torch.stack([t['input_ids'][0] for t in tokenized])
    attention_masks = torch.stack([t['attention_mask'][0] for t in tokenized])
    return input_ids, attention_masks

Y = data1['BinaryNumTarget']
X = data1.drop(['BinaryNumTarget'], axis=1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train_ids1, x_train_masks1 = tokenize_text(tokenizer1, x_train1['tweet'])
x_test_ids1, x_test_masks1 = tokenize_text(tokenizer1, x_test1['tweet'])

train_data1 = TensorDataset(x_train_ids1, x_train_masks1, torch.tensor(y_train1.values))
train_loader1 = DataLoader(train_data1, batch_size=16, shuffle=True)

test_data1 = TensorDataset(x_test_ids1, x_test_masks1, torch.tensor(y_test1.values))
test_loader1 = DataLoader(test_data1, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1 = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model1.to(device)
optimizer1 = torch.optim.AdamW(model1.parameters(), lr=5e-5)

train_losses1 = []
for epoch in range(3):  # Number of epochs
    model1.train()
    total_loss = 0
    for batch in tqdm(train_loader1):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer1.zero_grad()
        outputs = model1(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        labels = F.one_hot(labels.long(), num_classes=2).float()
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)
        loss.backward()
        optimizer1.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader1)
    train_losses1.append(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(torch.softmax(logits, dim=-1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels

preds1, labels1 = evaluate_model(model1, test_loader1)
accuracy1 = accuracy_score(labels1, preds1)
precision1 = precision_score(labels1, preds1)
recall1 = recall_score(labels1, preds1)
f1_1 = f1_score(labels1, preds1)

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy1:.2f}")
print(f"Precision: {precision1:.2f}")
print(f"Recall: {recall1:.2f}")
print(f"F1-Score: {f1_1:.2f}")

indices = random.sample(range(len(labels1)), 10)
print("\nRandom Samples (Test Data, Actual vs Predicted):")
for i in indices:
    sample_text = x_test1['tweet'].iloc[i]
    print(f"Tweet: {sample_text}")
    print(f"Actual: {labels1[i]}, Predicted: {preds1[i]}")
    print("-" * 50)


cm1 = confusion_matrix(labels1, preds1)
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses1) + 1), train_losses1, label="Training Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()