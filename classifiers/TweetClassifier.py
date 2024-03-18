import os
import time
import pandas as pd
import datasets as ds

from utils.utils import read_csv_file
from classifiers.LLMClassifier import LLMClassifier


class TweetClassifier(LLMClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # REALLLLY messed up paths - if I were to redesign this
    # move the data read and processing into the individual main functions
    # way I dealt with paths in here is HELLLLLA jank.
    def read_data(self, read_clean=False):
        if not read_clean:
            train = read_csv_file(f"{self.data_path}/train.csv")
            valid = read_csv_file(f"{self.data_path}/validation.csv")
            test = read_csv_file(f"{self.data_path}/test.csv")
        else:
            try:
                train = pd.read_pickle(f"{self.data_path}/train-clean.pkl")
                valid = pd.read_pickle(f"{self.data_path}/validation-clean.pkl")
                test = pd.read_pickle(f"{self.data_path}/test-clean.pkl")
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"{error}"
                    f"could not read file - make sure the path is correct."
                    f"current working dir: {os.getcwd()}"
                )

        return train, valid, test

    def preprocess_data(self, clean_file_exists):
        print("~~~~E~~~~")
        print(time.time())

        # read in files: for distribution and all of the data
        self.read_parse_distribution(
            human_filepath=f"{self.data_path}/human_tweet_parse_count.pkl",
            ai_filepath=f"{self.data_path}/bot_tweet_parse_count.pkl",
        )
        train, valid, test = self.read_data(clean_file_exists)

        print("~~~~F~~~~")
        print(time.time())

        if not clean_file_exists:
            # since we need 1's and 0's for training instead of text - this mapping needs to occur
            train["account.type"] = train["account.type"].map({"human": 0, "bot": 1})
            valid["account.type"] = valid["account.type"].map({"human": 0, "bot": 1})
            test["account.type"] = test["account.type"].map({"human": 0, "bot": 1})

            # ensuring the text is always less than 512 characters long
            # i think this is bc the Berkley nlp doesn't like it
            train = train[train.text.str.len() < 512]
            valid = valid[valid.text.str.len() < 512]
            test = test[test.text.str.len() < 512]

            # create the new parse column for the data
            train["parse"] = self.create_new_parse_col(train)
            valid["parse"] = self.create_new_parse_col(valid)
            test["parse"] = self.create_new_parse_col(test)

            # create the parse category column
            train["pcat"] = self.create_new_parse_category_col(train)
            valid["pcat"] = self.create_new_parse_category_col(valid)
            test["pcat"] = self.create_new_parse_category_col(test)

            # create a combined triple column of the text, the parse, and the category as a str
            train["concat"] = train["text"] + [" <s> "] + train["parse"] + [" <s> "] + train["pcat"].astype(str)
            valid["concat"] = valid["text"] + [" <s> "] + valid["parse"] + [" <s> "] + valid["pcat"].astype(str)
            test["concat"] = test["text"] + [" <s> "] + test["parse"] + [" <s> "] + test["pcat"].astype(str)

            # if you need the types account type - you can left join it back in on the text
            train = train.drop(
                ["screen_name", "class_type", "parse", "pcat"], axis=1
            ).rename(columns={"account.type": "labels"})
            valid = valid.drop(
                ["screen_name", "class_type", "parse", "pcat"], axis=1
            ).rename(columns={"account.type": "labels"})
            test = test.drop(
                ["screen_name", "class_type", "parse", "pcat"], axis=1
            ).rename(columns={"account.type": "labels"})

            train.to_pickle(f"{self.data_path}/train-clean.pkl")
            valid.to_pickle(f"{self.data_path}/valid-clean.pkl")
            test.to_pickle(f"{self.data_path}/test-clean.pkl")

        print("~~~~G~~~~")
        print(time.time())

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
