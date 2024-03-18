import os
import pandas as pd
import datasets as ds

from utils.utils import read_json_file, divide_data
from classifiers.LLMClassifier import LLMClassifier


class AbstractClassifier(LLMClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_data(self, clean_file_exists=False):
        if not clean_file_exists:
            human = read_json_file(f"{self.data_path}/ieee-init.jsonl")
            chatgpt = read_json_file(f"{self.data_path}/ieee-chatgpt-generation.jsonl")

            # drop unnecessary columns, rename columns, create new label column, and concat two dfs together
            human = human.drop(["id", "title", "keyword"], axis=1).rename(columns={"abstract": "text"})
            chatgpt = chatgpt.drop(["id", "title", "keyword"], axis=1).rename(columns={"abstract": "text"})
            human["labels"] = 0
            chatgpt["labels"] = 1
            cheat = pd.concat([chatgpt, human], axis=0)
            return divide_data(cheat)
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
        # read in files: for distribution and all of the data
        self.read_parse_distribution(
            human_filepath=f"{self.data_path}/human_abstract_parse_count.pkl",
            ai_filepath=f"{self.data_path}/bot_abstract_parse_count.pkl",
        )
        train, valid, test = self.read_data(clean_file_exists)

        if not clean_file_exists:
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
            train = train.drop(["parse", "pcat"], axis=1)
            valid = valid.drop(["parse", "pcat"], axis=1)
            test = test.drop(["parse", "pcat"], axis=1)

            train.to_pickle(f"{self.data_path}/train-clean.pkl")
            valid.to_pickle(f"{self.data_path}/valid-clean.pkl")
            test.to_pickle(f"{self.data_path}/test-clean.pkl")

        return ds.DatasetDict({
            "train": ds.Dataset.from_pandas(train),
            "valid": ds.Dataset.from_pandas(valid),
            "test": ds.Dataset.from_pandas(test)
        })

    def data_distribution(self, split="valid"):
        ...
