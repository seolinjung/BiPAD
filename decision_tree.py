import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from processing.process_images import scikit_preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

def ocr_features(ocrs):
    vectorizer = TfidfVectorizer(
        input = "content",
        token_pattern = r"(?u)[\w'\-:.,%]+"
    )
    features = vectorizer.fit_transform(ocrs)
    vectorizer.get_feature_names_out()
    return features

def main():
    raw_data = pd.read_csv(Path.cwd() / "data" / "final_ground_truth.csv")
    labels = raw_data["gt"]
    paths = raw_data["path"]
    ocrs = raw_data["ocr"]
    ocr_features = ocr_features(ocrs)
    


if __name__ == "__main__":
    main()