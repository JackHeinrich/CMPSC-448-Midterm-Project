import torch
import torch.nn as nn

import torch.nn.functional as F

from text_pipeline import get_data_loaders, PADDING_IDX

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

def main() -> None:

    # Getting training data set
    train_loader, val_loader, test_loader, vocab_size, num_labels = get_data_loaders(max_length=512, batch_size=32)
    

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

    first_batch = next(iter(train_loader))[0]

    print(f"[main] The model works, here are it's predictions for the first batch of training data: \n{cnn(first_batch)}")

if __name__ == "__main__":
    main()