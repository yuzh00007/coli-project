import pandas as pd


def read_csv_file(file_path, sep=";"):
    df = pd.read_csv(file_path, sep=sep)
    return df


def get_prediction(scores: dict):
    return max(scores, key=lambda x: x["score"])
