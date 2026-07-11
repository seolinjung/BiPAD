import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

de_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german")
en_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

def tokenize(data):
    tokens = []
    for row in data.itertuples():
        print(row)
        tokenized = de_tokenizer(row.ocr, return_tensors='pt') if row.region == 'DE' else en_tokenizer(row.ocr, return_tensors='pt')
        print(tokenized)
        tokens.append(tokenized)
    return tokens