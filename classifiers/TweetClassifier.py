import pandas as pd
import datasets as ds

from utils.utils import read_csv_file
from classifiers.LLMClassifier import LLMClassifier


class TweetClassifier(LLMClassifier):
    def __init__(
        self, base_model, tokenizer, seed=42, num_epochs=5, sample=None
    ):
        super(TweetClassifier, self).__init__(base_model, tokenizer, seed, num_epochs, sample)

    def read_data(self):
        train = read_csv_file("../data/tweepfake/train.csv")
        valid = read_csv_file("../data/tweepfake/validation.csv")
        test = read_csv_file("../data/tweepfake/test.csv")

        return train, valid, test

    def preprocess_data(self):
        train, valid, test = self.read_data()

        # ensuring the text is always less than 512 words long
        train = train[train.text.str.len() < 512]
        valid = valid[valid.text.str.len() < 512]
        test = test[test.text.str.len() < 512]

        # since we need 1's and 0's for training instead of text - this mapping needs to occur
        train["account.type"] = train["account.type"].map({"human": 0, "bot": 1})
        valid["account.type"] = valid["account.type"].map({"human": 0, "bot": 1})
        test["account.type"] = test["account.type"].map({"human": 0, "bot": 1})

        # if you need the types account type - you can left join it back in on the text
        train = train.drop(["screen_name", "class_type"], axis=1).rename(columns={"account.type": "labels"})
        valid = valid.drop(["screen_name", "class_type"], axis=1).rename(columns={"account.type": "labels"})
        test = test.drop(["screen_name", "class_type"], axis=1).rename(columns={"account.type": "labels"})

        return ds.DatasetDict({
            "train": ds.Dataset.from_pandas(train),
            "valid": ds.Dataset.from_pandas(valid),
            "test": ds.Dataset.from_pandas(test)
        })

    def data_distribution(self, split="valid"):
        dataset = self.datasets[split]
        human_count = dataset[dataset.class_type == "human"].text.count()
        rnn_count = dataset[dataset.class_type == "rnn"].text.count()
        oth_count = dataset[dataset.class_type == "others"].text.count()
        gpt2_count = dataset[dataset.class_type == "gpt2"].text.count()
        return pd.DataFrame({
            "type": ["human", "rnn", "markov", "gpt-2"],
            "count": [human_count, rnn_count, oth_count, gpt2_count]
        })
