from pathlib import Path

from classifiers.TweetClassifier import TweetClassifier
from classifiers.AbstractClassifier import AbstractClassifier


def run_abstract(
    model, tokenizer, sample_size=None, baseline=False, epoch=5, batch_size=8
):
    """
    :param model: the model
    :param tokenizer: the tokenizer
    :param sample_size: if you want to run with a small sample - for testing code
    :param baseline: if you want to evaluate off the shelf performance
    :param epoch: number of epochs to train
    :param batch_size: batch size for traning
    """
    classifier = AbstractClassifier(
        base_model=model,
        tokenizer=tokenizer,
        clean_file_exists=Path("./data/cheat/train-clean.pkl").exists(),
        finetune_with_parse=False,
        data_folder_path="./data/cheat",
        per_device_train_batch_size=batch_size
    )
    print(classifier.datasets)

    classifier_w_parse = AbstractClassifier(
        base_model=model,
        tokenizer=tokenizer,
        # second time - should be just read the data the first guy wrote to disk
        clean_file_exists=Path("./data/cheat/train-clean.pkl").exists(),
        finetune_with_parse=True,
        data_folder_path="./data/cheat",
        per_device_train_batch_size=batch_size
    )

    # do set up - in order to create all the things we will need during the finetune
    # phase. it's here to set up the trainer and evaluate
    classifier.finetune_setup(num_epochs=epoch, sample_size=sample_size)
    classifier_w_parse.finetune_setup(num_epochs=epoch, sample_size=sample_size)

    if baseline:
        print(f"validation results out of the box")
        print(classifier.evaluate("test", sample_size=sample_size))
        return

    print("-" * 15, "\n", "training finetuned model")
    classifier.train("./models/finetuned-abstract")

    print("\n", "test results after fine-tuning")
    print(classifier.evaluate("test", sample_size=sample_size))

    print("-" * 15, "\n", "training parse-tree model")
    classifier_w_parse.train("./models/parsed-abstract")

    print("\n", "test results after fine-tuning with parse trees")
    print(classifier_w_parse.evaluate("test", sample_size=sample_size))


def run_twitter(
    model, tokenizer, sample_size=None, baseline=False, epoch=5, batch_size=8
):
    """
    :param model: the model
    :param tokenizer: the tokenizer
    :param sample_size: if you want to run with a small sample - for testing code
    :param baseline: if you want to evaluate off the shelf performance
    :param epoch: number of epochs to train
    :param batch_size: batch size for traning
    """
    classifier = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer,
        clean_file_exists=Path("./data/tweepfake/train-clean.pkl").exists(),
        finetune_with_parse=False,
        data_folder_path="./data/tweepfake",
        per_device_train_batch_size=batch_size
    )
    print(classifier.datasets)

    classifier_w_parse = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer,
        # second time - should be just read the data the first guy wrote to disk
        clean_file_exists=Path("./data/tweepfake/train-clean.pkl").exists(),
        finetune_with_parse=True,
        data_folder_path="./data/tweepfake",
        per_device_train_batch_size=batch_size
    )

    # do set up - in order to create all the things we will need during the finetune
    # phase. it's here to set up the trainer and evaluate
    classifier.finetune_setup(num_epochs=epoch, sample_size=sample_size)
    classifier_w_parse.finetune_setup(num_epochs=epoch, sample_size=sample_size)

    if baseline:
        print(f"validation results out of the box")
        print(classifier.evaluate("test", sample_size=sample_size))
        return

    print("-" * 15, "\n", "training finetuned model")
    classifier.train("./models/finetuned-twitter")

    print("\n", "test results after fine-tuning")
    print(classifier.evaluate("test", sample_size=sample_size))

    print("-" * 15, "\n", "training parse-tree model")
    classifier_w_parse.train("./models/parsed-twitter")

    print("\n", "test results after fine-tuning with parse trees")
    print(classifier_w_parse.evaluate("test", sample_size=sample_size))
