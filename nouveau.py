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

def evaluate(test_indices, roberta_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        "./nouveau",
        num_labels=2
    )
    model.eval()
    input_ids, attention_mask, labels = [], [], []
    for i in test_indices:
        input_ids.append(roberta_dataset[i]['input_ids'])
        attention_mask.append(roberta_dataset[i]['attention_mask'])
        labels.append(roberta_dataset[i]['labels'])
    with torch.no_grad():
        logits = model(
            input_ids=torch.stack(input_ids),
            attention_mask=torch.stack(attention_mask)
        ).logits
    print(logits)

def train(train_indices, roberta_dataset):
    train_dataset = Subset(roberta_dataset, train_indices)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-large",
        num_labels=2
    ).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-05, weight_decay=0.00)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    for epoch in range(10):
        model.train()
        print(f"epoch #{epoch+1}")
        epoch_loss = 0
        for batch, data in enumerate(train_loader):
            logits = model(
                input_ids=data['input_ids'].to(device),
                attention_mask=data['attention_mask'].to(device)
            ).logits
            probabilities = torch.softmax(logits, dim=0)
            current_loss = loss_function(probabilities, data['labels'].to(device))
            current_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += current_loss.item()
            print(f"batch #{batch+1} - loss: {current_loss.item():.4f}")
        print(f"\nepoch #{epoch+1} out of {10} - loss: {epoch_loss/len(train_loader)}")

    torch.save(
        model.state_dict(), './test_model.pth'
    )

    return model

    # args = TrainingArguments(
    #     output_dir='./',
    #     save_strategy='no',
    #     learning_rate=1e-6,
    #     per_device_train_batch_size=32,
    #     num_train_epochs=1,
    #     weight_decay=0.00
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     train_dataset=Subset(full_dataset, train_indices)
    # )
