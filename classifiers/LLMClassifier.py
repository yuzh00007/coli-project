import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TextClassificationPipeline, get_scheduler

from utils.utils import get_prediction


class LLMClassifier:
    def __init__(self, base_model, tokenizer, seed=42):
        self.seed = seed
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = base_model
        self.tokeniser = tokenizer
        self.pipe = TextClassificationPipeline(model=base_model, tokenizer=tokenizer, return_all_scores=True)

        self.datasets = self.preprocess_data()

        # for finetuning
        self.optimizer = AdamW(self.model.parameters(), lr=5e-5)
        self.finetune_epoch = 3

    def read_data(self):
        ...

    def preprocess_data(self):
        ...

    def finetune(self):
        import time
        start = time.time()

        def tokenize_function(examples):
            return self.tokeniser(examples["text"], padding="max_length", truncation=True)

        tokenized_ds = self.datasets.map(tokenize_function, batched=True)

        tokenized_ds = tokenized_ds.remove_columns(["text"])
        small_train_dataset = tokenized_ds["train"].shuffle(seed=42).select(range(100))
        small_valid_dataset = tokenized_ds["valid"].shuffle(seed=42).select(range(100))

        train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=512)
        valid_dataloader = DataLoader(small_valid_dataset, batch_size=512)

        num_training_steps = self.finetune_epoch * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        self.model.to(self.device)

        import evaluate
        import numpy as np
        from tqdm.auto import tqdm

        progress_bar = tqdm(total=num_training_steps, desc="training steps")

        for epoch in range(self.finetune_epoch):
            self.model.train()
            for batch in train_dataloader:
                batch = {k: torch.from_numpy(np.asarray(v)).to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

            metric = evaluate.load("accuracy")
            self.model.eval()
            for batch in valid_dataloader:
                batch = {k: torch.from_numpy(np.asarray(v)).to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(**batch)

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                metric.add_batch(predictions=predictions, references=batch["labels"])

            return {
                "acc": metric.compute(),
                "time": time.time() - start
            }

    def fine_tune(self):
        def tokenize_function(examples):
            return self.tokeniser(examples["text"], padding="max_length", truncation=True)

        tokenized_ds = self.datasets.map(tokenize_function, batched=True)

        tokenized_ds = tokenized_ds.remove_columns(["text"])
        small_train_dataset = tokenized_ds["train"].shuffle(seed=42).select(range(100))
        small_valid_dataset = tokenized_ds["valid"].shuffle(seed=42).select(range(100))

        import numpy as np
        import evaluate

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        from transformers import TrainingArguments, Trainer
        training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=small_valid_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

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
