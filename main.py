import spacy
import benepar
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from classifiers.TweetClassifier import TweetClassifier


def main():
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")
    model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta")

    nlp = spacy.load('en_core_web_sm')
    benepar.download('benepar_en3')
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})
    spacy.prefer_gpu()

    classifier = TweetClassifier(
        base_model=model,
        tokenizer=tokenizer,
        nlp=nlp,
        sample=10
    )
    print(classifier.datasets)

    print("\n", "-" * 15)
    eval_results = classifier.evaluate("test", sample_size=10)
    print(f"validation results out of the box {eval_results}")

    classifier.finetune()

    print("\n", "-" * 15)
    eval_results = classifier.evaluate("test", sample_size=10)
    print(f"test results after fine-tuning {eval_results}")


if __name__ == "__main__":
    main()
