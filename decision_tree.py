import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from skimage.feature import hog
from sklearn.metrics import accuracy_score

from processing.process_images import scikit_preprocess

def format_data():
    raw_data = pd.read_csv(Path.cwd() / "data" / "final_ground_truth.csv")
    raw_data["ocr"] = raw_data["ocr"].replace(np.nan, "")
    return raw_data

def extract_ocr_features(ocrs):
    vectorizer = TfidfVectorizer(
        input = "content",
        token_pattern = r"(?u)[\w'\-:.,%]+"
    )
    features = vectorizer.fit_transform(ocrs)
    return features

def extract_image_features(paths, categories):
    features = []
    for i, path in enumerate(paths):
        complete_path = Path.cwd() / "data" / categories[i] / "adshots" / path
        feature = hog(
            image = scikit_preprocess(complete_path),
            feature_vector = True
        )
        features.append(feature)
    return np.array(features)

def predict_one(tree, ocr_feature):
    return tree.predict(ocr_feature)

def evaluate_all(tree, test_indices, full_dataset):
    test_data = full_dataset.data.iloc[test_indices]
    test_features = full_dataset.features[test_indices]
    predictions = tree.predict(test_features)
    return accuracy_score(test_data["gt"], predictions)

def evaluate_de(tree, de_indices, full_dataset):
    test_data = full_dataset.data.iloc[de_indices]
    test_features = full_dataset.features[de_indices]
    predictions = tree.predict(test_features)
    return accuracy_score(test_data["gt"], predictions)

def evaluate_us(tree, us_indices, full_dataset):
    test_data = full_dataset.data.iloc[us_indices]
    test_features = full_dataset.features[us_indices]
    predictions = tree.predict(test_features)
    return accuracy_score(test_data["gt"], predictions)

def train(train_indices, full_dataset):
    train_data = full_dataset.data.iloc[train_indices]
    train_features = full_dataset.features[train_indices]
    tree = DecisionTreeClassifier()
    tree.fit(train_features, train_data["gt"])
    return tree