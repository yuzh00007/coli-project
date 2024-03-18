import os
import spacy
import benepar
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def read_csv_file(file_path, sep=";"):
    try:
        df = pd.read_csv(file_path, sep=sep)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}"
            f"could not read file - make sure the path is correct."
            f"current working dir: {os.getcwd()}"
        )

    return df


def read_json_file(file_path):
    try:
        json_obj = pd.read_json(path_or_buf=file_path, lines=True)
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"{error}"
            f"could not read file - make sure the path is correct."
            f"current working dir: {os.getcwd()}"
        )

    return json_obj


def create_nlp_object():
    nlp = spacy.load('en_core_web_sm')
    benepar.download('benepar_en3')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    spacy.prefer_gpu()

    return nlp


def get_prediction(scores: dict):
    return max(scores, key=lambda x: x["score"])


def tokenize_function(examples, tokeniser, col_name):
    return tokeniser(examples[col_name], padding="max_length", truncation=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    return {"f1": f1, "recall": recall, "precision": precision, "accuracy": acc}


def combine_two_dicts(dict1: dict, dict2: dict):
    """
    combines two dicts into one, adding their values
    dictionaries must have numeral values
    :return: new dictionary with combined keys, and added values
    """
    new_dict = {}
    for key, value in dict1.items():
        d2_val = dict2.get(key)
        new_dict.update({
            key: value + d2_val if d2_val else value
        })

    for key, value in dict2.items():
        d1_val = dict1.get(key)
        new_dict.update({
            key: value + d1_val if d1_val else value
        })

    return new_dict


def divide_data(dataset):
    train, test = train_test_split(dataset, 0.15)
    train, valid = train_test_split(train, 0.2)
    return train, valid, test
