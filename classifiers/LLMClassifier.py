import torch
import functools
from transformers import TextClassificationPipeline, TrainingArguments, Trainer

from utils.parse_trees import generate_parse
from utils.utils import tokenize_function, compute_metrics


class LLMClassifier:
    def __init__(
        self,
        base_model, tokenizer, nlp,
        seed=42, num_epochs=5, sample=None
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

        self.datasets = self.preprocess_data()

        self.finetune_setup(num_epochs=num_epochs, seed=seed, sample_size=sample)

    def read_data(self):
        ...

    def preprocess_data(self):
        ...

    def create_new_parse_col(self, ds):
        parses = ds.apply(
            lambda x: ", ".join(
                [str(x) for x in generate_parse(self.nlp, x["text"], depth=3)]
            ), axis=1
        )
        return parses

    def finetune_setup(
        self, num_epochs=5, seed=42, sample_size=None
    ):
        """
        create all the necessary stuff for the finetune step
        including the evaluator, optimizer, arguments, and the trainer
        """
        tokenized_ds = self.datasets.map(
            functools.partial(tokenize_function, tokeniser=self.tokeniser),
            batched=True
        )

        tokenized_ds = tokenized_ds.remove_columns(["text"])
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

    def finetune(self):
        self.trainer.train()

    def calc_perplexity(self):
        ...

    def generate_parse_trees(self):
        ...

    def calc_parse_metric(self):
        ...

    def data_distribution(self):
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
