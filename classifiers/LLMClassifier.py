
import torch
import evaluate
import functools
from torch.optim import AdamW
from transformers import TextClassificationPipeline, TrainingArguments, Trainer

from utils.utils import get_prediction, tokenize_function, compute_metrics


class LLMClassifier:
    def __init__(self, base_model, tokenizer, seed=42):
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

        self.datasets = self.preprocess_data()

        # for finetuning
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.finetune_epoch = 10

    def read_data(self):
        ...

    def preprocess_data(self):
        ...

    def finetune(
        self, num_epochs=10, seed=42, sample_size=None
    ):
        tokenized_ds = self.datasets.map(
            functools.partial(tokenize_function, tokeniser=self.tokeniser),
            batched=True
        )

        tokenized_ds = tokenized_ds.remove_columns(["text"])
        small_train_dataset = tokenized_ds["train"]
        small_valid_dataset = tokenized_ds["valid"]

        # seeded run
        if seed:
            small_train_dataset = small_train_dataset.shuffle(seed=seed)
            small_valid_dataset = small_valid_dataset.shuffle(seed=seed)
        # sample run
        if sample_size:
            small_train_dataset = small_train_dataset.select(range(sample_size))
            small_valid_dataset = small_valid_dataset.select(range(sample_size))

        training_args = TrainingArguments(
            output_dir="test_trainer",
            evaluation_strategy="epoch",
            num_train_epochs=num_epochs,
        )

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_valid_dataset,
            compute_metrics=compute_metrics,
            logging_dir='./logs',
        )
        self.trainer.train()

    def calc_perplexity(self):
        ...

    def generate_parse_trees(self):
        ...

    def calc_parse_metric(self):
        ...

    def data_distribution(self):
        ...

    def evaluate(self, split="test"):
        import time
        dataset_x = list(self.datasets[split]["text"])
        dataset_y = list(self.datasets[split]["labels"])

        start = time.time()

        predictions = [
            get_prediction(x) for x in self.pipe(dataset_x)
        ]
        # it's label here b/c that's what the model returns - doesn't need to match labels like our dataset
        pred_edit = [0 if x["label"] == "Human" else 1 for x in predictions]

        end = time.time()

        return {
            "time": end - start,
            "accuracy": sum(x == y for x, y in zip(dataset_y, pred_edit)) / len(dataset_x)
        }
