import os
import torch
import functools
import pandas as pd
from transformers import TextClassificationPipeline, TrainingArguments, Trainer

from utils.utils import tokenize_function, compute_metrics
from utils.parse_trees import generate_parse, generate_freq_category


class LLMClassifier:
    def __init__(
        self,
        base_model, tokenizer, data_folder_path,
        seed=42, clean_file_exists=False,
        finetune_with_parse=False,
    ):
        self.seed = seed
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = base_model
        self.model.to(self.device)
        self.tokeniser = tokenizer
        self.pipe = TextClassificationPipeline(
            model=base_model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=self.device
        )
        self.trainer = ...

        self.finetune_with_parse = finetune_with_parse

        self.data_path = data_folder_path
        self.datasets = self.preprocess_data(clean_file_exists)
        self.tokenise_dataset()
        self.parse_distrib = ...

    def read_data(self):
        # to be filled in by subclasses
        ...

    def read_parse_distribution(self, human_filepath, ai_filepath):
        try:
            human_distrib = pd.read_pickle(human_filepath)
            ai_distrib = pd.read_pickle(ai_filepath)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{error}"
                f"could not read file - make sure the path is correct."
                f"current working dir: {os.getcwd()}"
            )

        self.parse_distrib = generate_freq_category(human_distrib)

    def preprocess_data(self, clean_file_exists):
        # to be filled in by subclasses
        ...

    def create_new_parse_col(self, ds):
        parses = ds.apply(
            lambda x: ", ".join(
                [str(x) for x in generate_parse(x["text"], depth=3)]
            ), axis=1
        )
        return parses

    def create_new_parse_category_col(self, ds):
        parse_cat = ds.apply(
            lambda x: [
                self.parse_distrib.get(parse)
                if self.parse_distrib.get(parse) else 0
                for parse in x["parse"].split(", ")
            ],
            axis=1
        )
        return parse_cat

    def tokenise_dataset(self):
        """
        turn a specific column (depending on finetune_with_parse argument from constructor)
        into input_ids for the Trainer
        """
        col_name = "text"
        if self.finetune_with_parse:
            col_name = "concat"
        self.datasets = self.datasets.map(
            functools.partial(tokenize_function, tokeniser=self.tokeniser, col_name=col_name),
            batched=True
        )

    def finetune_setup(
        self, num_epochs=5, seed=42, sample_size=None
    ):
        """
        create all the necessary stuff for the finetune step
        including the evaluator, optimizer, arguments, and the trainer
        """
        train_dataset = self.datasets["train"]
        valid_dataset = self.datasets["valid"]

        # seeded run
        if seed:
            train_dataset = train_dataset.shuffle(seed=seed)
            valid_dataset = valid_dataset.shuffle(seed=seed)
        # sample run
        if sample_size:
            train_dataset = train_dataset.select(range(sample_size))
            valid_dataset = valid_dataset.select(range(sample_size))

        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            num_train_epochs=num_epochs,
        )
        training_args.set_optimizer(name="adamw_torch", learning_rate=1e-3)
        training_args.set_lr_scheduler(name="constant_with_warmup", warmup_ratio=0.05)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics,
        )

    def train(self, model_path: str = None):
        self.model.train()
        self.trainer.train()
        if model_path:
            self.trainer.save_model(model_path)

    def calc_perplexity(self):
        # TODO - use library LM-PPL or huggingface evaluate's perplexity calculator
        ...

    def evaluate(self, split="test", sample_size=None):
        ds = self.datasets[split]
        if sample_size:
            ds = ds.select(range(sample_size))

        self.model.eval()
        return self.trainer.evaluate(ds)
