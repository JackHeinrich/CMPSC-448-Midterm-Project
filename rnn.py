import torch
import torch.nn as nn

from torch.nn.utils.rnn import pack_padded_sequence

from text_pipeline import get_data_loaders, get_cross_domain_data_loaders, PADDING_IDX

from training import train_model, evaluate

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

# Build an RNN with the settings used for every experiment. embedding_dim and dropout match the CNN for a fair comparison.
def build_rnn(vocab_size: int, num_labels: int) -> RNN:

    # Initialization specifications
    hidden_size = 128

    embedding_dim = 300

    rnn = RNN(embedding_dim=embedding_dim)

    rnn.initialize_embedding(num_embeddings=vocab_size, padding_idx=PADDING_IDX)

    # Create a single bidirectional LSTM layer
    rnn.initialize_lstm_layer(hidden_size=hidden_size, num_layers=1, bidirectional=True)

    rnn.initialize_dropout_layer(p=0.5)

    rnn.initialize_output_layer(num_labels)

    return rnn

# Train an RNN on one condition, then test it once on the test set (research questions 1 and 2).
def evaluate_rnn_on_condition(condition, training_epochs, training_domain: str | None = None, testing_domain: str | None = None) -> tuple[float, list[int], list[int], list[str]]:

    # Seed for replicability
    torch.manual_seed(1)

    # Getting training data set
    train_loader, val_loader, test_loader, vocab_size, num_labels, label_names = get_data_loaders(condition, train_domain=training_domain, test_domain=testing_domain)

    rnn = train_model(build_rnn(vocab_size, num_labels), train_loader, val_loader, training_epochs, model_name="rnn", condition=condition, training_domain=training_domain or "all")

    print(f"[Main] Beginning final test.")
    # Final test
    final_accuracy, predictions, actual_answers = evaluate(model=rnn, eval_loader=test_loader)
    print(f"[Main] final test complete. Final test accuracy: {final_accuracy}")

    return final_accuracy, predictions, actual_answers, label_names

# Train an RNN once on training_domain, then test it on every domain's test questions (research question 3).
# Returns {testing domain: (accuracy, predictions, gold labels)} and the label names.
def evaluate_rnn_across_domains(condition, training_epochs, training_domain: str) -> tuple[dict[str, tuple[float, list[int], list[int]]], list[str]]:

    # Seed for replicability
    torch.manual_seed(1)

    train_loader, val_loader, test_loaders, vocab_size, num_labels, label_names = get_cross_domain_data_loaders(condition, training_domain)

    rnn = train_model(build_rnn(vocab_size, num_labels), train_loader, val_loader, training_epochs, model_name="rnn", condition=condition, training_domain=training_domain)

    results = {}

    for testing_domain, test_loader in test_loaders.items():
        results[testing_domain] = evaluate(model=rnn, eval_loader=test_loader)
        print(f"[Main] Trained on {training_domain}, tested on {testing_domain}, accuracy: {results[testing_domain][0]}")

    return results, label_names

def main() -> None:
    evaluate_rnn_on_condition("outputs", 10)

if __name__ == "__main__":
    main()
