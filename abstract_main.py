from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from classifiers.AbstractClassifier import AbstractClassifier


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

    classifier = AbstractClassifier(
        base_model=model,
        tokenizer=tokenizer,
        clean_file_exists=Path("./data/cheat/train-clean.pkl").exists(),
        finetune_with_parse=False,
        data_folder_path="./data/cheat"
    )
    print(classifier.datasets)

    classifier_w_parse = AbstractClassifier(
        base_model=model,
        tokenizer=tokenizer,
        # second time - should be just read the data the first guy wrote to disk
        clean_file_exists=Path("./data/cheat/train-clean.pkl").exists(),
        finetune_with_parse=True,
        data_folder_path="./data/cheat"
    )

    # do set up - in order to create all the things we will need during the finetune
    # phase. it's here to set up the trainer and evaluate
    classifier.finetune_setup(num_epochs=5)
    classifier_w_parse.finetune_setup(num_epochs=5)

    # print(f"validation results out of the box")
    # print(classifier.evaluate("test"))

    print("-" * 15, "\n", "training finetuned model")
    classifier.train("./models/finetuned-abstract")

    print("\n", "test results after fine-tuning")
    print(classifier.evaluate("test"))

    print("-" * 15, "\n", "training parse-tree model")
    classifier_w_parse.train("./models/parsed-abstract")

    print("\n", "test results after fine-tuning with parse trees")
    print(classifier_w_parse.evaluate("test"))


if __name__ == "__main__":
    import time
    start = time.time()
    main()
    print(f"total runtime: {time.time() - start:.2f}")
