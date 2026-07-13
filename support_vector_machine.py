import numpy as np

from sklearn import svm
from sklearn.metrics import f1_score, precision_score, recall_score

from processing.dataset import AdsDataset
from processing.extract_features import extract_tfidf, extract_hog

def predict_one(tree, ocr_feature):
    return tree.predict(ocr_feature)

def get_metrics(gt, predictions):
    return {
        "f1": f1_score(gt, predictions),
        "precision": precision_score(gt, predictions),
        "recall": recall_score(gt, predictions)
    }

def evaluate(machine, test_indices, full_dataset):
    test_data = full_dataset.data.iloc[test_indices]
    test_features = full_dataset.features[test_indices]
    predictions = machine.predict(test_features)
    return get_metrics(test_data["gt"], predictions)

def train(train_indices, full_dataset):
    train_data = full_dataset.data.iloc[train_indices]
    train_features = full_dataset.features[train_indices]
    machine = svm.SVC()
    machine.fit(train_features, train_data["gt"])
    return machine

def run(root, data, train_indices, test_indices_de, test_indices_us):
    test_indices = test_indices_de + test_indices_us
    text_features = extract_tfidf(data["ocr"].replace(np.nan, ""))
    # image_features = extract_hog(data["path"], data["subcategory"])
    full_dataset = AdsDataset(root, data, text_features)
    # full_image_dataset = AdsDataset(root, data, image_features)
    machine = train(train_indices, full_dataset)
    # image_machine = train(train_indices, full_image_dataset)
    results_all = evaluate(machine, test_indices)
    results_de = evaluate(machine, test_indices_de)
    results_us = evaluate(machine, test_indices_us)
    print(results_all)
    print(results_de)
    print(results_us)