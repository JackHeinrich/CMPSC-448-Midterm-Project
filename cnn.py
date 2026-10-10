import torch
import torch.nn as nn

import torch.nn.functional as F

from text_pipeline import get_data_loaders, get_cross_domain_data_loaders, PADDING_IDX

from training import train_model, evaluate

class CNN(nn.Module):

    def __init__(self, embedding_dim) -> None:
        super().__init__()
        # embedding_dim is used by embedding layer and conv layer so it is stored as a instance attribute.
        self.embedding_dim = embedding_dim
        self.conv_layers = nn.ModuleList()

    def initialize_embedding(self, num_embeddings, padding_idx) -> None:
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.embedding_dim, padding_idx=padding_idx)

    def add_conv_layer(self, num_filters, kernel_size) -> None:
        new_layer = nn.Conv1d(in_channels=self.embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        self.conv_layers.append(new_layer)

    def initialize_output_layer(self, num_classes) -> None:

        in_features = 0

        # In features = sum of filters of each layer
        for layer in self.conv_layers:
            in_features += layer.out_channels

        self.output_layer = nn.Linear(in_features=in_features, out_features=num_classes)

    # Dropout layer is meant to reduce over fitting by randomly removing p% the features on every batch.
    def initialize_dropout_layer(self, p) -> None:
        self.dropout = nn.Dropout(p=p)

    # Forward implementation adapted from Chris Tran's "A Complete Guide to CNN for Sentence Classificatoin with PyTorch" recommended by a TA.
    def forward(self, input_ids) -> torch.Tensor:

        # X_embed are token embeddings (batch_size, max_length, embedding_dim)
        x_embed = self.embedding(input_ids)

        # Reshape embeddings to shape (batch size, embed_dim, max_len) since conv1d expects channels before length.
        x_reshaped = x_embed.transpose(2, 1)

        # Apply CNN and ReLu
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv_layers]

        # Max pooling (take max across each filter's strongest score)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2]) for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed fully connected layer
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)

        # Scores of each label for whole batch.
        logits = self.output_layer(self.dropout(x_fc))

        return logits

# Build a CNN with the settings used for every experiment.
def build_cnn(vocab_size: int, num_labels: int) -> CNN:

    # Initialization specifications
    num_filters = 100

    embedding_dim = 300

    cnn = CNN(embedding_dim=embedding_dim)

    cnn.initialize_embedding(num_embeddings=vocab_size, padding_idx=PADDING_IDX)

    # Create 3 conv layers with window size 3, 4, and 5.
    for i in range(3,6):
        cnn.add_conv_layer(num_filters=num_filters, kernel_size=i)

    cnn.initialize_dropout_layer(p=0.5)

    cnn.initialize_output_layer(num_labels)

    return cnn

# Train a CNN on one condition, then test it once on the test set (research questions 1 and 2).
def evaluate_cnn_on_condition(condition, training_epochs, training_domain: str | None = None, testing_domain: str | None = None) -> tuple[float, list[int], list[int], list[str]]:

    # Seed for replicability
    torch.manual_seed(1)

    # Getting training data set
    train_loader, val_loader, test_loader, vocab_size, num_labels, label_names = get_data_loaders(condition, train_domain=training_domain, test_domain=testing_domain)

    cnn = train_model(build_cnn(vocab_size, num_labels), train_loader, val_loader, training_epochs)

    print(f"[Main] Beginning final test.")
    # Final test
    final_accuracy, predictions, actual_answers = evaluate(model=cnn, eval_loader=test_loader)
    print(f"[Main] final test complete. Final test accuracy: {final_accuracy}")

    return final_accuracy, predictions, actual_answers, label_names

# Train a CNN once on training_domain, then test it on every domain's test questions (research question 3).
# Returns {testing domain: (accuracy, predictions, gold labels)} and the label names.
def evaluate_cnn_across_domains(condition, training_epochs, training_domain: str) -> tuple[dict[str, tuple[float, list[int], list[int]]], list[str]]:

    # Seed for replicability
    torch.manual_seed(1)

    train_loader, val_loader, test_loaders, vocab_size, num_labels, label_names = get_cross_domain_data_loaders(condition, training_domain)

    cnn = train_model(build_cnn(vocab_size, num_labels), train_loader, val_loader, training_epochs)

    results = {}

    for testing_domain, test_loader in test_loaders.items():
        results[testing_domain] = evaluate(model=cnn, eval_loader=test_loader)
        print(f"[Main] Trained on {training_domain}, tested on {testing_domain}, accuracy: {results[testing_domain][0]}")

    return results, label_names

def main() -> None:
    evaluate_cnn_on_condition("outputs", 1)

if __name__ == "__main__":
    main()
