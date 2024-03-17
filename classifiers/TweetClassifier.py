import pandas as pd
import datasets as ds

from utils.utils import read_csv_file
from classifiers.LLMClassifier import LLMClassifier


class TweetClassifier(LLMClassifier):
    def __init__(
        self, base_model, tokenizer, nlp, seed=42, num_epochs=5
    ):
        super(TweetClassifier, self).__init__(base_model, tokenizer, nlp, seed, num_epochs)

    def read_data(self):
        train = read_csv_file("../data/tweepfake/train.csv")
        valid = read_csv_file("../data/tweepfake/validation.csv")
        test = read_csv_file("../data/tweepfake/test.csv")

        return train, valid, test

    def preprocess_data(self):
        # read in files: for distribution and all of the data
        self.read_parse_distribution(
            human_filepath="../data/human_tweet_parse_count.pkl",
            ai_filepath="../data/bot_tweet_parse_count.pkl",
        )
        train, valid, test = self.read_data()

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
        train = train[train.parses.str.len() > 1]
        valid["parse"] = self.create_new_parse_col(valid)
        valid = valid[valid.parses.str.len() > 1]
        test["parse"] = self.create_new_parse_col(test)

        # create the parse category column
        train["pcat"] = self.create_new_parse_category_col(train)
        valid["pcat"] = self.create_new_parse_category_col(valid)
        test["pcat"] = self.create_new_parse_category_col(test)

        # create a combined triple column of the text, the parse, and the category as a str
        train["concat"] = train["text"] + [" <s> "] + train["parses"] + [" <s> "] + train["pcat"].astype(str)
        valid["concat"] = valid["text"] + [" <s> "] + valid["parses"] + [" <s> "] + valid["pcat"].astype(str)
        test["concat"] = test["text"] + [" <s> "] + test["parses"] + [" <s> "] + test["pcat"].astype(str)

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
