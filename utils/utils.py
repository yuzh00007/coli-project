import evaluate
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_file(file_path, sep=";"):
    df = pd.read_csv(file_path, sep=sep)

    return df


def read_json_file(file_path):
    json_obj = pd.read_json(path_or_buf=file_path, lines=True)

    return json_obj


def get_prediction(scores: dict):
    return max(scores, key=lambda x: x["score"])


def tokenize_function(examples, tokeniser, col_name):
    return tokeniser(examples[col_name], padding="max_length", truncation=True)


def compute_metrics(eval_pred, f1: evaluate.Metric = None):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # in case you want to provide a specific metric
    if f1:
        return f1.compute(predictions=predictions, references=labels)

    # otherwise, default to "glue-mrpc", which will give accuracy and f1
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    acc = evaluate.load("accuracy")
    return {
        "f1": f1.compute(predictions=predictions, references=labels),
        "recall": recall.compute(predictions=predictions, references=labels),
        "precision": precision.compute(predictions=predictions, references=labels),
        "accuracy": acc.compute(predictions=predictions, references=labels),
    }


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
