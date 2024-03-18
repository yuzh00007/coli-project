import pandas as pd
import datasets as ds

from utils.utils import read_csv_file
from classifiers.LLMClassifier import LLMClassifier


class TweetClassifier(LLMClassifier):
    def __init__(
        self, base_model, tokenizer, nlp, seed=42, clean_file_exists=False
    ):
        super(TweetClassifier, self).__init__(base_model, tokenizer, nlp, seed, clean_file_exists)

    def read_data(self, read_clean=False):
        if not read_clean:
            train = read_csv_file("../data/tweepfake/train.csv")
            valid = read_csv_file("../data/tweepfake/validation.csv")
            test = read_csv_file("../data/tweepfake/test.csv")
        else:
            train = pd.read_pickle("../data/tweepfake/train-clean.pkl")
            valid = pd.read_pickle("../data/tweepfake/validation-clean.pkl")
            test = pd.read_pickle("../data/tweepfake/test-clean.pkl")

        return train, valid, test

    def preprocess_data(self, clean_file_exists):
        # read in files: for distribution and all of the data
        self.read_parse_distribution(
            human_filepath="../data/human_tweet_parse_count.pkl",
            ai_filepath="../data/bot_tweet_parse_count.pkl",
        )
        train, valid, test = self.read_data(clean_file_exists)

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
            # if a text fails to parse, we remove it from the data
            # except for the test dataset, which we will keep all the data in there
            train["parse"] = self.create_new_parse_col(train)
            train = train[train.parse.str.len() > 1]
            valid["parse"] = self.create_new_parse_col(valid)
            valid = valid[valid.parse.str.len() > 1]
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

            train.to_pickle("../data/tweepfake/train-clean.csv")
            valid.to_pickle("../data/tweepfake/valid-clean.csv")
            test.to_pickle("../data/tweepfake/test-clean.csv")

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
