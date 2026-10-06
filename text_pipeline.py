import re
from collections import Counter

from data_splitter import load_splits

# Minimum amount of occurances of a token to include it in the vocab.
MIN_FREQ = 2

# Break strings up into arrays of tokens. These are lowercase individual words broken up by whitespace and punctuation. Punctuation is kept as its own token.
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())

# Build a vocab by indexing each word that appeared more than MIN_FREQ times in the training set.
def build_vocab(training_texts) -> dict[str, int]:

    word_counts = Counter()

    # Count occurances of each token
    for text in training_texts:
        word_counts.update(tokenize(text))

    # Init vocab with pad and unknown tokens necessary for training.
    # Pad tokens ensure that every sequence in a batch has the same length.
    # Unk tokens are used for words that appear in the validation or test set but not in the training set.

    vocab = {"<pad>": 0, "<unk>": 1}

    for word, count in word_counts.most_common():
        if count >= MIN_FREQ:
            vocab[word] = len(vocab)
    return vocab


# Test that it works.
def main():
    train_df, val_df, test_df = load_splits()
    vocab = build_vocab(train_df["Answer"])

    print("Train vocab:", vocab)

if __name__ == "__main__":
    main()