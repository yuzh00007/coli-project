from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from utils.utils import create_nlp_object
from classifiers.TweetClassifier import TweetClassifier

import os


def main():
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta",)
    # i noticed that the from_pretrained was taking minutes to do
    # it's half a gigabyte of tensors
    # so I simply saved the original model and read it locally each time
    try:
        model = AutoModelForSequenceClassification.from_pretrained("./models/original")
    except OSError:
        model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta",)
        model.save_pretrained("./models/original", from_pt=True)
    nlp = create_nlp_object()

    clean_file = Path("./data/tweepfake/train-clean.pkl")
    clean_file_exist = clean_file.exists()
    classifier = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer,
        nlp=nlp,
        clean_file_exists=clean_file_exist,
        finetune_with_parse=False,
    )
    print(classifier.datasets)

    classifier_w_parse = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer,
        nlp=nlp,
        clean_file_exists=clean_file_exist,
        finetune_with_parse=True
    )

    # do set up - in order to create all the things we will need during the finetune
    # phase. it's here to set up the trainer and evaluate
    classifier.finetune_setup(num_epochs=5, sample_size=10)
    classifier_w_parse.finetune_setup(num_epochs=5, sample_size=10)

    print(f"validation results out of the box")
    print(classifier.evaluate("test", sample_size=10))

    print("\n", "-" * 15)
    classifier.train("./models/finetuned-twitter")

    print("\n", "-" * 15)
    print(f"test results after fine-tuning")
    print(classifier.evaluate("test", sample_size=10))

    print("\n", "-" * 15)
    classifier_w_parse.train("./models/parsed-twitter")

    print("\n", "-" * 15)
    print(f"test results after fine-tuning with parses")
    print(classifier_w_parse.evaluate("test", sample_size=10))


if __name__ == "__main__":
    # since we do the run from inside /sic_cluster
    # this is a hammer-meet-nail method to resolve dumb issues with file paths
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)

    main()
