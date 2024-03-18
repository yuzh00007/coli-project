from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from utils.utils import create_nlp_object
from classifiers.TweetClassifier import TweetClassifier


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

    print(clean_file_exist)

    classifier = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer,
        nlp=nlp,
        clean_file_exists=clean_file_exist,
        finetune_with_parse=False,
        data_folder_path="./data/tweepfake"
    )
    print(classifier.datasets)


if __name__ == "__main__":
    main()
