import copy
import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence

from text_pipeline import get_data_loaders, PADDING_IDX

from training import train_one_epoch, evaluate, DEVICE

class RNN(nn.Module):

    def __init__(self, embedding_dim) -> None:
        super().__init__()
        # embedding_dim is used by embedding layer and lstm layer so it is stored as a instance attribute.
        self.embedding_dim = embedding_dim

    # padding_idx is stored so forward can count how many real (non padding) tokens each sequence has.
    def initialize_embedding(self, num_embeddings, padding_idx) -> None:
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=self.embedding_dim, padding_idx=padding_idx)

    # LSTM reads the embeddings one token at a time, updating its hidden state after each token.
    # Bidirectional runs a second LSTM that reads the sequence backwards, so the end of the answer is summarized as well as the start.
    def initialize_lstm_layer(self, hidden_size, num_layers=1, bidirectional=True) -> None:
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional)

    def initialize_output_layer(self, num_classes) -> None:

        # In features = final hidden state of each direction concatenated
        in_features = self.lstm.hidden_size
        if self.lstm.bidirectional:
            in_features *= 2

        self.output_layer = nn.Linear(in_features=in_features, out_features=num_classes)

    # Dropout layer is meant to reduce over fitting by randomly removing p% the features on every batch.
    def initialize_dropout_layer(self, p) -> None:
        self.dropout = nn.Dropout(p=p)

    def forward(self, input_ids) -> torch.Tensor:

        # X_embed are token embeddings (batch_size, max_length, embedding_dim)
        x_embed = self.embedding(input_ids)

        # Real length of each sequence. Padding is only added to the end, so counting non padding tokens gives the length. Clamped to 1 since packing can't handle empty sequences.
        lengths = (input_ids != self.padding_idx).sum(dim=1).clamp(min=1)

        # Pack the batch so the LSTM stops at each sequence's real length instead of reading the padding tokens.
        # Without this, short answers would end with hundreds of padding steps that wash out the final hidden state.
        x_packed = pack_padded_sequence(x_embed, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # h_n is the final hidden state of every layer and direction (num_layers * num_directions, batch_size, hidden_size)
        _, (h_n, _) = self.lstm(x_packed)

        # Use the last layer's final hidden state. For bidirectional, concatenate the forward direction (h_n[-2]) and backward direction (h_n[-1]).
        if self.lstm.bidirectional:
            x_fc = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            x_fc = h_n[-1]

        # Scores of each label for whole batch.
        logits = self.output_layer(self.dropout(x_fc))

        return logits

def main() -> None:

    # Seed for replicability
    torch.manual_seed(1)

    # Getting training data set
    train_loader, val_loader, test_loader, vocab_size, num_labels = get_data_loaders(condition="outputs")

    # Initialization specifications. embedding_dim, dropout, optimizer, and epochs match the CNN for a fair comparison.
    hidden_size = 128

    embedding_dim = 300

    training_epochs = 10

    rnn = RNN(embedding_dim=embedding_dim)

    rnn.initialize_embedding(num_embeddings=vocab_size, padding_idx=PADDING_IDX)

    # Create a single bidirectional LSTM layer
    rnn.initialize_lstm_layer(hidden_size=hidden_size, num_layers=1, bidirectional=True)

    rnn.initialize_dropout_layer(p=0.5)

    rnn.initialize_output_layer(num_labels)

    # Move all weights to the GPU if available. Must happen before creating the optimizer.
    rnn.to(DEVICE)

    print(f"[Main] Training on device: {DEVICE}")

    #  Soft max loss function (-log(correct class probability))
    loss_function = nn.CrossEntropyLoss()

    # Optimizer used for updated weights using Adam to pick learning rate for each weight.
    optimizer = torch.optim.Adam(params=rnn.parameters())

    # Track best accuracy to use weights with best validation performance preventing overfitting
    best_epoch_accuracy = 0

    best_weights = None

    best_epoch = None

    # Train, then evaluate against validate data set
    for i in range(training_epochs):
        epoch_num = i + 1
        print(f"[Main] Training epoch {epoch_num}")
        epoch_loss = train_one_epoch(model=rnn, train_loader=train_loader, optimizer=optimizer, loss_function=loss_function)
        print(f"[Main] Finished training epoch {epoch_num}, loss: {epoch_loss}")
        print(f"[Main] Validating after epoch {epoch_num}")
        accuracy, _, _ = evaluate(model=rnn, eval_loader=val_loader)

        # Update best epoch weights
        if accuracy > best_epoch_accuracy:
            best_epoch_accuracy = accuracy
            best_weights = copy.deepcopy(rnn.state_dict())
            best_epoch = epoch_num

        print(f"[Main] Validated after epoch {epoch_num}, accuracy: {accuracy}")

    # Load weights with best validation performance
    rnn.load_state_dict(best_weights)

    print(f"[Main] Finished training for {training_epochs} epochs")
    print(f"[Main] best epoch: {best_epoch}, best validation accuracy: {best_epoch_accuracy}")
    print(f"[Main] Beginning final test.")
    # Final test
    final_accuracy, _, _ = evaluate(model=rnn, eval_loader=test_loader)
    print(f"[Main] final test complete. Final test accuracy: {final_accuracy}")


if __name__ == "__main__":
    main()
