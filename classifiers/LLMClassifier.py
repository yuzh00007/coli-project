import torch
import functools
import pandas as pd
from transformers import TextClassificationPipeline, TrainingArguments, Trainer

from utils.parse_trees import generate_parse, generate_freq_category
from utils.utils import tokenize_function, compute_metrics


class LLMClassifier:
    def __init__(
        self,
        base_model, tokenizer, nlp,
        seed=42, clean_file_exists=False
    ):
        self.seed = seed
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.nlp = nlp
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

        self.datasets = self.preprocess_data(clean_file_exists)
        self.parse_distrib = ...

    def read_data(self):
        # to be filled in by subclasses
        ...

    def read_parse_distribution(self, human_filepath, ai_filepath):
        human_distrib = pd.read_pickle(human_filepath)
        ai_distrib = pd.read_pickle(ai_filepath)

        self.parse_distrib = generate_freq_category(human_distrib)

    def preprocess_data(self, clean_file_exists):
        # to be filled in by subclasses
        ...

    def create_new_parse_col(self, ds):
        parses = ds.apply(
            lambda x: ", ".join(
                [str(x) for x in generate_parse(self.nlp, x["text"], depth=3)]
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

    def finetune_setup(
        self, num_epochs=5, seed=42, sample_size=None, finetune_with_parse=False
    ):
        """
        create all the necessary stuff for the finetune step
        including the evaluator, optimizer, arguments, and the trainer
        """
        col_name = "text"
        if finetune_with_parse:
            col_name = "concat"
        tokenized_ds = self.datasets.map(
            functools.partial(tokenize_function, tokeniser=self.tokeniser, col_name=col_name),
            batched=True
        )

        tokenized_ds = tokenized_ds.remove_columns(["text", "concat"])
        train_dataset = tokenized_ds["train"]
        valid_dataset = tokenized_ds["valid"]

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

    def train(self):
        self.trainer.train()

    def calc_perplexity(self):
        # TODO - use library LM-PPL or huggingface evaluate's perplexity calculator
        ...

    def evaluate(self, split="test", sample_size=None):
        """
        evaluates the dataset with our trainer
        :param split: either "train", "valid", or "test"
        :param sample_size: provide a value if you want to run on smaller eval size
        """
        eval_set = self.datasets[split]
        if sample_size:
            eval_set = eval_set.select(range(20))

        self.trainer.eval_dataset = eval_set
        self.trainer.evaluate()
