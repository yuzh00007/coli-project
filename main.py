from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from classifiers.TweetClassifier import TweetClassifier


def main():
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")

    classifier = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer
    )

    print(classifier.datasets)

    eval_results = classifier.evaluate("valid")
    print(eval_results)

    print(classifier.fine_tune())


if __name__ == "__main__":
    main()
