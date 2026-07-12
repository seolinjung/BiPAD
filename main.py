import yaml
import argparse
from pathlib import Path
import json

import yaml
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import AutoTokenizer
from processing.dataset import AdsDataset, RobertaAdsDataset
from processing.process_data import get_region_indices_test, get_region_indices_train
from processing.clean_data import sync_data
from processing.extract_features import extract_tfidf, extract_hog
import decision_tree
import support_vector_machine
import nouveau
import evaluate

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sota", required=False, action="store_true")
    parser.add_argument("--dt", required=False, action="store_true")
    parser.add_argument("--svm", required=False, action="store_true")
    parser.add_argument("--nouveau", required=False, action="store_true")
    return parser.parse_args()

def main(args):
    root = Path.cwd()
    data = pd.read_csv(root / "data" / "cleaned_data.csv")
    data["ocr"] = data["ocr"].replace(np.nan, "")
    sota_results = pd.read_csv(root / "prior" / "llm_annotation.csv")
    test_indices_de, test_indices_us = get_region_indices_test(root, data)
    train_indices_de, train_indices_us = get_region_indices_train(data, test_indices_de, test_indices_us)
    test_indices = test_indices_de + test_indices_us
    train_indices = train_indices_de + train_indices_us
    if args.sota:
        print(evaluate.evaluate_all(test_indices, sota_results, data))
        print(evaluate.evaluate_sota_de(test_indices_de, sota_results, data))
        print(evaluate.evaluate_sota_us(test_indices_us, sota_results, data))
    if args.dt:
        text_features = extract_tfidf(data)
        # image_features = extract_hog(data["path"], data["subcategory"])
        full_dataset = AdsDataset(root, data, text_features)
        # full_image_dataset = AdsDataset(root, data, image_features)
        tree = decision_tree.train(train_indices, full_dataset)
        # tree = decision_tree.train(train_indices_de, full_dataset)
        # image_tree = decision_tree.train(train_indices, full_image_dataset)
        print(decision_tree.evaluate_de(tree, test_indices_de, full_dataset))
        print(decision_tree.evaluate_us(tree, test_indices_us, full_dataset))
    if args.svm:
        # text_features = extract_tfidf(data["ocr"].replace(np.nan, ""))
        image_features = extract_hog(data["path"], data["subcategory"])
        # full_dataset = AdsDataset(root, data, text_features)
        full_image_dataset = AdsDataset(root, data, image_features)
        # machine = support_vector_machine.train(train_indices, full_dataset)
        image_machine = support_vector_machine.train(train_indices, full_image_dataset)
        print(support_vector_machine.evaluate_de(image_machine, test_indices_de, full_image_dataset))
        print(support_vector_machine.evaluate_us(image_machine, test_indices_us, full_image_dataset))
    if args.nouveau:
        de_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german")
        en_tokenizer = AutoTokenizer.from_pretrained("FacebookAI/xlm-roberta-large")
        tokens = nouveau.tokenize(data, de_tokenizer, en_tokenizer)
        print(tokens)
        roberta_dataset = RobertaAdsDataset(root, data, tokens)
        full_dataset = AdsDataset(root, data, tokens)
        nouveau.train(train_indices, roberta_dataset)
        nouveau.evaluate_de(test_indices, roberta_dataset, full_dataset)

if __name__ == "__main__":
    args = get_args()
    main(args)