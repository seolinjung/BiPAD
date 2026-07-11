import numpy as np
import pandas as pd
from pathlib import Path

from sklearn import svm
from sklearn.metrics import accuracy_score

def predict_one(tree, ocr_feature):
    return tree.predict(ocr_feature)

def evaluate_all(machine, test_indices, full_dataset):
    test_data = full_dataset.data.iloc[test_indices]
    test_features = full_dataset.features[test_indices]
    predictions = machine.predict(test_features)
    return accuracy_score(test_data["gt"], predictions)

def evaluate_de(machine, de_indices, full_dataset):
    test_data = full_dataset.data.iloc[de_indices]
    test_features = full_dataset.features[de_indices]
    predictions = machine.predict(test_features)
    return accuracy_score(test_data["gt"], predictions)

def evaluate_us(machine, us_indices, full_dataset):
    test_data = full_dataset.data.iloc[us_indices]
    test_features = full_dataset.features[us_indices]
    predictions = machine.predict(test_features)
    return accuracy_score(test_data["gt"], predictions)

def train(train_indices, full_dataset):
    train_data = full_dataset.data.iloc[train_indices]
    train_features = full_dataset.features[train_indices]
    machine = svm.SVC()
    machine.fit(train_features, train_data["gt"])
    return machine