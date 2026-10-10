import re

from collections import Counter
from typing import Literal

import torch
from torch.utils.data import DataLoader, TensorDataset

from data_splitter import load_splits

# Text pipeline adapted from Chris Tran's "A Complete Guide to CNN for Sentence Classification with PyTorch"

# Padding index exported to keep consistent across nn implementations

PADDING_IDX = 0

# Minimum amount of occurances of a token to include it in the vocab since single instances don't help the model learn.
MIN_FREQ = 2

# Break strings up into arrays of tokens. These are lowercase individual words broken up by whitespace and punctuation. Punctuation is kept as its own token.
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())

# Build a vocab by indexing each word that appeared more than MIN_FREQ times in the training set.
def build_vocab(training_texts) -> dict[str, int]:

    print(f"[build_vocab] Building vocab from {len(training_texts)} training texts")

    word_counts = Counter()

    # Count occurances of each token
    for text in training_texts:
        word_counts.update(tokenize(text))

    # Init vocab with pad and unknown tokens necessary for training.
    # Pad tokens ensure that every sequence in a batch has the same length.
    # Unk tokens are used for words that appear in the validation or test set but not in the training set.

    vocab = {"<pad>": PADDING_IDX, "<unk>": 1}

    for word, count in word_counts.most_common():
        if count >= MIN_FREQ:
            vocab[word] = len(vocab)

    print(f"[build_vocab] Finished building vocab with {len(vocab)} tokens")

    return vocab

# Pad each sentence to the max sentence length and encode tokens to their index in vocab.
def encode(tokenized_text: list[str], vocab: dict[str, int], max_length: int) -> list[int]:

    print(f"[encode] Encoding {len(tokenized_text)} tokenized sentences to max length {max_length}")

    # Encoded sentences as a list of integers in place of vocab tokens.
    input_ids = []

    for tokenized_sentence in tokenized_text:

        # Truncate to ensure all strings are the same length. Pad for same reason
        tokenized_sentence = tokenized_sentence[:max_length]
        tokenized_sentence += ['<pad>'] * (max_length - len(tokenized_sentence))

        # Replace each token with its index or unk if not in vocab.
        input_id = [vocab.get(token) if vocab.get(token) is not None else vocab['<unk>'] for token in tokenized_sentence]
        
        input_ids.append(input_id)

    print(f"[encode] Finished encoding {len(tokenized_text)} tokenized sentences to max length {max_length}")

    return input_ids

# Generic function for getting all 3 data loaders to be used with pytorch neural networks. 
def get_data_loaders(max_length: int, batch_size: int = 50) -> tuple[DataLoader, DataLoader, DataLoader, int, int]:

    print(f"[get_data_loaders] Getting data loaders, max length: {max_length}, batch size: {batch_size}")

    train_df, val_df, test_df = load_splits()

    vocab = build_vocab(train_df["Answer"])

    # Assign each source to a integer
    label_ids = {name: i for i, name in enumerate(sorted(train_df["Source"].unique()))}

    # Helper for creating data loaders for teh 3 splits
    def get_data_loader(df, should_shuffle = False) -> DataLoader:
        encoded_texts = encode(tokenized_text=[tokenize(text) for text in df["Answer"]], vocab=vocab, max_length=max_length)
        
        #Create a 2D tensor for the (len(encoded_texts), max_length) inputs and a 1D tensor for the (len(encoded_texts)) labels. 
        inputs = torch.tensor(encoded_texts)
        labels = torch.tensor([label_ids[source] for source in df["Source"]])
        
        data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=batch_size, shuffle=should_shuffle)
        return data_loader

    train_data_loader = get_data_loader(train_df, True)

    val_data_loader = get_data_loader(val_df)

    test_data_loader = get_data_loader(test_df)

    vocab_size = len(vocab)

    num_labels = len(label_ids)

    print(f"[get_data_loaders] Finished getting data loaders for all splits, vocab size: {vocab_size}, num labels: {num_labels}")

    # Expose vocab size and number of labels for model creation.
    return train_data_loader, val_data_loader, test_data_loader, vocab_size, num_labels

# Test that it works.
def main():
    print(get_data_loaders(max_length=50, batch_size=50))

if __name__ == "__main__":
    main()