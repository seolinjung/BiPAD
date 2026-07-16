from sklearn import svm

from dataset import AdsDataset
from processing.extract_features import extract_tfidf, extract_hog
from processing.process_data import get_metrics, print_metrics

def predict_one(tree, ocr_feature):
    return tree.predict(ocr_feature)

def train(train_indices, full_dataset):
    train_data = full_dataset.data.iloc[train_indices]
    train_features = full_dataset.features[train_indices]
    machine = svm.SVC()
    machine.fit(train_features, train_data["gt"])
    return machine

def evaluate(machine, test_indices, full_dataset):
    test_data = full_dataset.data.iloc[test_indices]
    test_features = full_dataset.features[test_indices]
    predictions = machine.predict(test_features)
    return get_metrics(test_data["gt"], predictions)

def run(root, data, train_indices, test_indices_de, test_indices_us):
    test_indices = test_indices_de + test_indices_us
    text_features = extract_tfidf(data["ocr"])
    # image_features = extract_hog(data["path"], data["subcategory"])
    full_dataset = AdsDataset(root, data, text_features)
    # full_image_dataset = AdsDataset(root, data, image_features)
    machine = train(train_indices, full_dataset)
    # image_machine = train(train_indices, full_image_dataset)
    results_all = evaluate(machine, test_indices, full_dataset)
    results_de = evaluate(machine, test_indices_de, full_dataset)
    results_us = evaluate(machine, test_indices_us, full_dataset)
    print("\n~Support Vector Machine Results~")
    print_metrics(results_all, results_de, results_us)