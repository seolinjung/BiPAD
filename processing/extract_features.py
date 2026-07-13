import numpy as np
from pathlib import Path

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from skimage.feature import hog

from transformers import AutoTokenizer

from processing.process_images import scikit_preprocess

def extract_tokens(data):
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
    tokens = []
    for row in data.itertuples():
        tokenized = tokenizer(
            row.ocr,
            truncation=True,
            padding="max_length",
            max_length=256
        )
        tokens.append(tokenized)
    return tokens

def extract_tfidf(ocrs):
    vectorizer = TfidfVectorizer(
        input = "content",
        token_pattern = r"(?u)[\w'\-:.,%]+"
    )
    features = vectorizer.fit_transform(ocrs)
    return features

def extract_hog(paths, categories):
    features = []
    for i, path in enumerate(paths):
        complete_path = Path.cwd() / "data" / categories[i] / "adshots" / path
        feature = hog(
            image = scikit_preprocess(complete_path),
            feature_vector = True
        )
        features.append(feature)
    return np.array(features)