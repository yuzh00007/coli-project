import os
import spacy
import benepar
import datasets
import numpy as np
import pandas as pd
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


def get_prediction(scores: dict):
    return max(scores, key=lambda x: x["score"])


def tokenize_function(examples, tokeniser, col_name):
    return tokeniser(examples[col_name], padding="max_length", truncation=True)


metric = datasets.load_metric('accuracy')


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    # acc = accuracy_score(labels, predictions)
    # precision, recall, f1, _ = precision_recall_fscore_support(
    #     labels, predictions, average="weighted"
    # )

    # return {"f1": f1, "recall": recall, "precision": precision, "accuracy": acc}

    return metric.compute(predictions=predictions, references=labels)


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


def divide_data(df, train_cut=.8, valid_cut=.9):
    """
    divides a df into three: train, valid, test
    train_cut .8 and valid_cut .9 => 80% 10% 10%
    train_cut .6 and valid_cut .8 => 60% 20% 20%

    :param df: single df to divide into 3 splits
    :param train_cut: what percent to make a cut in the data to divide train and rest
    :param valid_cut: what percent to make a cut in the data to divide test and rest
    :return: 3 dataframes
    """
    train, validate, test = np.split(
        df.sample(frac=1, random_state=1),
        [int(train_cut * len(df)), int(valid_cut * len(df))]
    )
    return train, validate, test
