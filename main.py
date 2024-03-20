from transformers import AutoTokenizer, AutoModelForSequenceClassification

from runners import run_abstract, run_twitter


def main(sample_size, baseline, epoch, batch_size):
    tokenizer = AutoTokenizer.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", )
    model = AutoModelForSequenceClassification.from_pretrained("Hello-SimpleAI/chatgpt-detector-roberta", num_labels=2)

    run_twitter(model, tokenizer, sample_size=sample_size, baseline=baseline, epoch=epoch, batch_size=batch_size)
    # run_abstract(model, tokenizer, sample_size=sample_size, baseline=baseline, epoch=epoch, batch_size=batch_size)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--baseline",
        help="provide a value to this and we'll run baseline, otherwise we'll run training",
        type=bool
    )
    parser.add_argument(
        "-s", "--sample_size",
        help="if you want to run with a subset of data (for testing, will be used for all three splits)",
        type=int
    )
    parser.add_argument(
        "-e", "--epoch",
        help="number of epochs to run",
        type=int
    )
    parser.add_argument(
        "-p", "--per_device_train_batch_size",
        help="number of epochs to run",
        type=int
    )
    parser.add_argument(
        "-l", "--learning_rate",
        help="what to set learning rate",
        type=float
    )
    args = parser.parse_args()

    import time
    start = time.time()
    main(
        sample_size=args.sample_size,
        baseline=args.baseline,
        epoch=args.epoch,
        batch_size=args.per_device_train_batch_size
    )
    print(f"total runtime: {time.time() - start:.2f}")
