from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from classifiers.TweetClassifier import TweetClassifier


def main():
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")

    classifier = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer
    )

    # print("data distribution across the splits:")
    # print(classifier.data_distribution(split="train"))
    # print(classifier.data_distribution(split="valid"))
    # print(classifier.data_distribution(split="test"))

    # print(classifier.datasets)
    # eval_results = classifier.evaluate("valid")
    # print(eval_results)

    print(classifier.fine_tune())

if __name__ == "__main__":
    main()
