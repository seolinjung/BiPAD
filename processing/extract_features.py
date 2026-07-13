import numpy as np
from pathlib import Path
from PIL import Image

from sklearn.feature_extraction.text import TfidfVectorizer
from skimage.feature import hog

from transformers import AutoTokenizer

def process_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize(size=(224,224))
    return img

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
            image = process_image(complete_path),
            feature_vector = True
        )
        features.append(feature)
    return np.array(features)