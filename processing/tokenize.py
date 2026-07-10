import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

de_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german")
en_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

def tokenize(full_dataset):
    tokens = []
    for idx in len(full_dataset):
        ocr = data.loc[idx, 'ocr']
        region = data.loc[idx, 'region']
        tokenized = de_tokenizer(ocr, return_tensors='pt') if region == 'DE' else en_tokenizer(ocr, return_tensors='pt')
    return tokenized

def tokenize