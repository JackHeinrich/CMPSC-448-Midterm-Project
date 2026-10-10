import copy
import torch
import torch.nn as nn

import torch.nn.functional as F

from text_pipeline import get_data_loaders, PADDING_IDX

from training import train_one_epoch, evaluate, get_device

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

def evaluate_cnn_on_condition(condition, training_epochs) -> tuple[float, list[int], list[int], list[str]]:

    # Seed for replicability
    torch.manual_seed(1)

    # Getting training data set
    train_loader, val_loader, test_loader, vocab_size, num_labels, label_names = get_data_loaders(condition)

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

    # Try move device to gpu if possible
    cnn.to(get_device())

    #  Soft max loss function (-log(correct class probability))
    loss_function = nn.CrossEntropyLoss()

    # Optimizer used for updated weights using Adam to pick learning rate for each weight.
    optimizer = torch.optim.Adam(params=cnn.parameters())

    # Track best accuracy to use weights with best validation performance preventing overfitting
    best_epoch_accuracy = 0

    best_weights = None

    best_epoch = None

    # Train, then evaluate against validate data set
    for i in range(training_epochs):
        epoch_num = i + 1
        print(f"[Main] Training epoch {epoch_num}")
        epoch_loss = train_one_epoch(model=cnn, train_loader=train_loader, optimizer=optimizer, loss_function=loss_function)
        print(f"[Main] Finished training epoch {epoch_num}, loss: {epoch_loss}")
        print(f"[Main] Validating after epoch {epoch_num}")
        accuracy, _, _ = evaluate(model=cnn, eval_loader=val_loader)

        # Update best epoch weights
        if accuracy > best_epoch_accuracy:
            best_epoch_accuracy = accuracy
            best_weights = copy.deepcopy(cnn.state_dict())
            best_epoch = epoch_num

        print(f"[Main] Validated after epoch {epoch_num}, accuracy: {accuracy}")

    # Load weights with best validation performance
    cnn.load_state_dict(best_weights)

    print(f"[Main] Finished training for {training_epochs} epochs")
    print(f"[Main] best epoch: {best_epoch}, best validation accuracy: {best_epoch_accuracy}")
    print(f"[Main] Beginning final test.")
    # Final test
    final_accuracy, predictions, actual_answers = evaluate(model=cnn, eval_loader=test_loader)
    print(f"[Main] final test complete. Final test accuracy: {final_accuracy}")

    return final_accuracy, predictions, actual_answers, label_names
    

def main() -> None:
    evaluate_cnn_on_condition("outputs", 1)

if __name__ == "__main__":
    main()