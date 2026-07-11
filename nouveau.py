import torch
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import Subset

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

def train(train_indices, full_dataset):
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/xlm-roberta-large",
        num_labels=2
    )
    args = TrainingArguments(
        output_dir='./',
        save_strategy='no',
        learning_rate=1e-6,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.00
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Subset(full_dataset, train_indices)
    )

    trainer.train()
    trainer.save_model('./nouveau')
