import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from skimage.feature import hog
from torch.utils.data import Subset, Dataset

from processing.process_images import scikit_preprocess
from processing.process_data import get_region_based_data_split

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

def evaluate(tree, test_indices, full_dataset):
    test_data = full_dataset.data.iloc[test_indices]
    test_ocr = test_data["ocr"].replace(np.nan, "")
    test_paths = test_data["path"]
    test_categories = test_data["categories"]
    test_features = extract_ocr_features(test_ocr)
    for idx in test_indices:
        print(predict_one(tree, test_featires[]))

def train(train_indices, full_dataset):
    train_data = full_dataset.data.iloc[train_indices]
    labels, paths, categories, ocrs = train_data["gt"], train_data["path"], train_data["subcategory"], train_data["ocr"].replace(np.nan, "")
    ocr_features = extract_ocr_features(ocrs)
    # image_features = extract_image_features(paths, categories)
    tree = tree.DecisionTreeClassifier()
    tree = tree.fit()
    # print(image_features)
    return tree