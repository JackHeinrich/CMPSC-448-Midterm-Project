import re

from collections import Counter
from typing import Callable, Literal

import torch
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd

from data_splitter import load_splits

from data_pipeline import STEM, SOCIAL_SCIENCE, HUMANITIES

# Text pipeline adapted from Chris Tran's "A Complete Guide to CNN for Sentence Classification with PyTorch"

# Padding index exported to keep consistent across nn implementations

PADDING_IDX = 0

# Minimum amount of occurances of a token to include it in the vocab since single instances don't help the model learn.
MIN_FREQ = 2

# Token used to seperate the question and answer when training on "both"
SEP_TOKEN = "<sep>"

# Dynamic max length depending on what condition we are training on
MAX_LENGTHS = {
        "inputs": 32,
        "outputs": 512,
        "both": 528
    }

# Constant batch size to keep comparison between CNN and RNN fair.
BATCH_SIZE = 32

# Break strings up into arrays of tokens. These are lowercase individual words broken up by whitespace and punctuation. Punctuation is kept as its own token.
def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())

# Build a vocab by indexing each word that appeared more than MIN_FREQ times in the training set.
def build_vocab(tokenized_vocab_building_set) -> dict[str, int]:

    print(f"[build_vocab] Building vocab from {len(tokenized_vocab_building_set)} training texts")

    word_counts = Counter()

    # Count occurances of each token
    for tokenized_text in tokenized_vocab_building_set:
        word_counts.update(tokenized_text)

    # Init vocab with pad and unknown tokens necessary for training.
    # Pad tokens ensure that every sequence in a batch has the same length.
    # Unk tokens are used for words that appear in the validation or test set but not in the training set.

    vocab = {"<pad>": PADDING_IDX, "<unk>": 1}

    for word, count in word_counts.most_common():
        if count >= MIN_FREQ:
            vocab[word] = len(vocab)

    print(f"[build_vocab] Finished building vocab with {len(vocab)} tokens")

    return vocab

# Tokenize a dfs outputs, inputs, or both into one list of tokenized texts.
# answer_transform optionally changes each tokenized answer, used by research question 4 to remove a signal (like punctuation) before training.
def tokenize_for_condition(df, condition: Literal["outputs", "inputs", "both"], answer_transform: Callable[[list[str]], list[str]] | None = None) -> list[list[str]]:
    
    # Stores the tokenized list for each questoin
    question_token_lists = []

    # Stores tokenized list for each answer
    answer_token_lists = []

    # Tokenize values and store all
    for text in df["Question"]:
            question_token_lists.append(tokenize(text))
    
    for text in df["Answer"]:
        answer_token_lists.append(tokenize(text))

    if answer_transform is not None:
        answer_token_lists = [answer_transform(answer_tokens) for answer_tokens in answer_token_lists]

    vocab_building_tokenized_texts = None

    if condition == "outputs":
        # Keep tokenized answers
        vocab_building_tokenized_texts = answer_token_lists
    elif condition == "inputs":
        # Keep only tokenized inputs
        vocab_building_tokenized_texts = question_token_lists
    elif condition == "both":
        # Concatenate each tokenized question with it's tokenized answer seperated with a SEP token
        concatenated_lists = []
        for i in range(len(question_token_lists)):
            concatenated_lists.append(question_token_lists[i] + [SEP_TOKEN] + answer_token_lists[i])
        vocab_building_tokenized_texts = concatenated_lists
    else:
        raise ValueError(f"Unknown condition {condition!r}. Expected 'outputs', 'inputs', or 'both'.")

    return vocab_building_tokenized_texts

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

# Domains used for research question 3, in the order they are reported.
DOMAINS = [STEM, SOCIAL_SCIENCE, HUMANITIES]

# Raise an error if domain is not one of DOMAINS.
def check_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}. Expected '{STEM}', '{SOCIAL_SCIENCE}', or '{HUMANITIES}'.")

# Build the vocab from the training set and create data loaders for training, validation, and each named test set.
def build_data_loaders(condition: Literal["outputs", "inputs", "both"], train_df: pd.DataFrame, val_df: pd.DataFrame, test_dfs: dict[str, pd.DataFrame], answer_transform: Callable[[list[str]], list[str]] | None = None) -> tuple[DataLoader, DataLoader, dict[str, DataLoader], int, int, list[str]]:

    max_length = MAX_LENGTHS[condition]

    # Build vocab based off of condition
    vocab = build_vocab(tokenize_for_condition(train_df, condition, answer_transform))

    # Assign each source to a integer
    label_names = sorted(train_df["Source"].unique())
    
    label_ids = {name: i for i, name in enumerate(label_names)}

    # Helper for creating data loaders for teh 3 splits
    def get_data_loader(df, should_shuffle = False) -> DataLoader:
        encoded_texts = encode(tokenized_text=tokenize_for_condition(df, condition, answer_transform), vocab=vocab, max_length=max_length)
        
        #Create a 2D tensor for the (len(encoded_texts), max_length) inputs and a 1D tensor for the (len(encoded_texts)) labels. 
        inputs = torch.tensor(encoded_texts)
        labels = torch.tensor([label_ids[source] for source in df["Source"]])
        
        data_loader = DataLoader(TensorDataset(inputs, labels), batch_size=BATCH_SIZE, shuffle=should_shuffle)
        return data_loader

    train_data_loader = get_data_loader(train_df, True)

    val_data_loader = get_data_loader(val_df)

    # One test loader per named test set
    test_data_loaders = {name: get_data_loader(test_df) for name, test_df in test_dfs.items()}

    vocab_size = len(vocab)

    num_labels = len(label_ids)

    print(f"[build_data_loaders] Finished getting data loaders for all splits, vocab size: {vocab_size}, num labels: {num_labels}")

    # Expose vocab size and number of labels for model creation.
    return train_data_loader, val_data_loader, test_data_loaders, vocab_size, num_labels, label_names

# Generic function for getting all 3 data loaders to be used with pytorch neural networks. 
def get_data_loaders(condition: Literal["outputs", "inputs", "both"], train_domain: str | None = None, test_domain: str | None = None, answer_transform: Callable[[list[str]], list[str]] | None = None) -> tuple[DataLoader, DataLoader, DataLoader, int, int, list[str]]:

    print(f"[get_data_loaders] Getting data loaders, max length: {MAX_LENGTHS[condition]}, batch size: {BATCH_SIZE}, condition: {condition}, training domain: {train_domain}, testing domain: {test_domain}")

    train_df, val_df, test_df = load_splits()

    # If train domain, pick only training examples and val examples where their domain is in training domain
    if train_domain is not None:
        check_domain(train_domain)
        train_df = train_df[train_df["domain"] == train_domain]
        val_df = val_df[val_df["domain"] == train_domain]

    # If test domain, take only test examples where domain is in test domain
    if test_domain is not None:
        check_domain(test_domain)
        test_df = test_df[test_df["domain"] == test_domain]

    train_data_loader, val_data_loader, test_data_loaders, vocab_size, num_labels, label_names = build_data_loaders(condition, train_df, val_df, {"test": test_df}, answer_transform)

    return train_data_loader, val_data_loader, test_data_loaders["test"], vocab_size, num_labels, label_names

# Data loaders for research question 3. Training and validation only use train_domain, and there is one test loader per domain
# so a model trained once can be tested on every domain. Test sets are each domain's usual test questions, so in-domain and cross-domain accuracy are measured on the same questions.
def get_cross_domain_data_loaders(condition: Literal["outputs", "inputs", "both"], train_domain: str) -> tuple[DataLoader, DataLoader, dict[str, DataLoader], int, int, list[str]]:

    print(f"[get_cross_domain_data_loaders] Getting data loaders, max length: {MAX_LENGTHS[condition]}, batch size: {BATCH_SIZE}, condition: {condition}, training domain: {train_domain}")

    check_domain(train_domain)

    train_df, val_df, test_df = load_splits()

    train_df = train_df[train_df["domain"] == train_domain]
    val_df = val_df[val_df["domain"] == train_domain]

    test_dfs = {domain: test_df[test_df["domain"] == domain] for domain in DOMAINS}

    return build_data_loaders(condition, train_df, val_df, test_dfs)

# Test that it works.
def main():
    print(get_data_loaders(condition="both"))

if __name__ == "__main__":
    main()