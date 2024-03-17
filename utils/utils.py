import evaluate
import numpy as np
import pandas as pd


def read_csv_file(file_path, sep=";"):
    df = pd.read_csv(file_path, sep=sep)

    return df


def get_prediction(scores: dict):
    return max(scores, key=lambda x: x["score"])


def tokenize_function(examples, tokeniser):
    return tokeniser(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred, metric: evaluate.Metric = None):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # in case you want to provide a specific metric
    if metric:
        return metric.compute(predictions=predictions, references=labels)

    # otherwise, default to "glue-mrpc", which will give accuracy and f1
    metric = evaluate.load("glue", "mrpc")
    return metric.compute(predictions=predictions, references=labels)
