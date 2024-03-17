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

    print("\n", "-" * 15)
    eval_results = classifier.evaluate("valid")
    print(f"validation results out of the box {eval_results}")

    classifier.finetune(
        num_epochs=5,
    )

    print("\n", "-" * 15)
    eval_results = classifier.evaluate("test")
    print(f"test results after fine-tuning {eval_results}")


if __name__ == "__main__":
    main()
