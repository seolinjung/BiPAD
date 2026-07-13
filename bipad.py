import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Subset, DataLoader

from dataset import RobertaAdsDataset
from processing.extract_features import extract_tokens
from processing.process_data import get_metrics, print_metrics

def train(root, data, roberta_dataset, train_indices, device):

    np.random.seed(42)
    np.random.shuffle(np.array(train_indices))

    split = int(0.8 * len(train_indices))
    train_only_indices = train_indices[:split]
    val_only_indices = train_indices[split:]

    root / "bipad"
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
    model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/xlm-roberta-large",num_labels=2).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-06, weight_decay=0.01)
    epochs = 80
    
    train_dataset = Subset(roberta_dataset, train_only_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )

    print(f"\n~Starting training~")

    for epoch in range(epochs):
        model.train()
        print(f"\nepoch no. {epoch+1}")
        epoch_loss = 0
        for batch, data in enumerate(train_loader):
            logits = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device)
            ).logits
            current_loss = loss_function(logits, data['labels'].to(device))
            current_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += current_loss.item()
            print(f"  batch no. {batch+1} - loss: {current_loss.item():.4f}")
        print(f"\nepoch no. {epoch+1}/{epochs} - loss: {epoch_loss/len(train_loader):.4f}")
        print(f"\n~Starting validation~")
        val_results = evaluate(model, roberta_dataset, val_only_indices, device)
        df = pd.DataFrame([val_results]).round(3)
        df.index = [""]
        df = df.T
        print("\n")
        print(df)

    model.save_pretrained(root / "bipad")
    tokenizer.save_pretrained(root / "bipad")

    return model

def evaluate(model, roberta_dataset, test_indices, device):
    test_dataset = Subset(roberta_dataset, test_indices)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False
    )
    all_predictions = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for _, data in enumerate(test_loader):
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

    return get_metrics(all_labels, all_predictions)

def run(root, data, train_indices, test_indices_de, test_indices_us, mode, device):
    tokens = extract_tokens(data)
    roberta_dataset = RobertaAdsDataset(root, data, tokens)
    test_indices = test_indices_de + test_indices_us
    
    if mode=="train":
        train(root, data, roberta_dataset, train_indices, device)
    if mode=="evaluate":
        model_path = root / "bipad"
        if not model_path.is_dir():
            print("\nCould not locate model folder. Aborting...")
            return
        model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        print("\nStarting evaluation...")
        results_all = evaluate(model, roberta_dataset, test_indices, device)
        results_de = evaluate(model, roberta_dataset, test_indices_de, device)
        results_us = evaluate(model, roberta_dataset, test_indices_us, device)
        print_metrics(results_all, results_de, results_us)