import torch
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Subset, DataLoader
import numpy as np
import torch.nn as nn
import torch.optim as optim

def tokenize(data, de_tokenizer, en_tokenizer):
    tokens = []
    for row in data.itertuples():
        tokenizer = de_tokenizer if row.region == 'DE' else en_tokenizer
        tokenized = tokenizer(
            row.ocr,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tokens.append(tokenized)
    return tokens

def evaluate(model, test_indices, roberta_dataset, device):
    test_dataset = Subset(roberta_dataset, test_indices)
    # model = torch.load("test_model.pth", map_location=device)
    # loss_function = nn.CrossEntropyLoss()
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=True
    )
    all_predictions = []
    all_labels = []
    model.eval()
    print(f"\nstarting testing")
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            logits = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device)
            ).logits
            probabilities = torch.softmax(logits, dim=0)
            predictions = torch.argmax(probabilities, dim=-1)
            all_predictions.extend(predictions)
            all_labels.extend(data['labels'])

    all_predictions = torch.stack(all_predictions).cpu().numpy()
    all_labels = torch.stack(all_labels).cpu().numpy()
    print(f1_score(all_labels, all_predictions))

def train(train_indices, roberta_dataset, device):
    train_dataset = Subset(roberta_dataset, train_indices)
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-large",
        num_labels=2
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-06, weight_decay=0.00)
    epochs = 80
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    print(f"\nstarting fine-tuning")
    for epoch in range(epochs):
        model.train()
        print(f"\nepoch no. {epoch+1}")
        epoch_loss = 0
        for batch, data in enumerate(train_loader):
            logits = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device)
            ).logits
            # probabilities = torch.softmax(logits, dim=0)
            current_loss = loss_function(logits, data['labels'].to(device))
            current_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += current_loss.item()
            print(f"  batch no. {batch+1} - loss: {current_loss.item():.4f}")
        print(f"\nepoch no. {epoch+1}/{epochs} - loss: {epoch_loss/len(train_loader):.4f}")

    torch.save(
        model.state_dict(), './test_model.pth'
    )

    return model