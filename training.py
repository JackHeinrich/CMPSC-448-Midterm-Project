import copy
import csv
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Train on the GPU if one with CUDA is available, otherwise fall back to the CPU. Models must be moved here with model.to(DEVICE) before training.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Per epoch training loss, validation accuracy, and time for every training run.
TRAINING_HISTORY_PATH = "results/training_history.csv"

TRAINING_HISTORY_HEADER = ["model", "condition", "training domain", "epoch", "training loss", "validation accuracy", "epoch seconds"]

# One row per training run with model size, best epoch, and total training time.
TRAINING_SUMMARY_PATH = "results/training_summary.csv"

TRAINING_SUMMARY_HEADER = ["model", "condition", "training domain", "parameters", "epochs", "best epoch", "best validation accuracy", "total seconds", "device"]

# Start both training logs over with just their headers. Called at the start of a full run so results from earlier runs aren't mixed in.
def reset_training_logs() -> None:
    for path, header in [(TRAINING_HISTORY_PATH, TRAINING_HISTORY_HEADER), (TRAINING_SUMMARY_PATH, TRAINING_SUMMARY_HEADER)]:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

# Add one row to a training log, writing the header first if the file doesn't exist yet.
def append_training_log(path: str, header: list[str], row: list) -> None:
    is_new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new_file:
            writer.writerow(header)
        writer.writerow(row)

# Do one pass through the training data set and update weights
def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: nn.CrossEntropyLoss) -> float:

    # Training mode turns dropout on
    model.train()

    total_loss = 0

    # Loop over every batch
    for inputs, labels in train_loader:

        # Move batch to the same device as the model
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        # Reset weights stored gradient
        optimizer.zero_grad()

        # Get models prediction
        logits = model(inputs)

        # Calculate loss for this batch
        loss = loss_function(logits, labels)

        # Backpropagate to get gradients for all weights
        loss.backward()

        # Update weights using gradients from backpropagation
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Return average loss
    return total_loss / len(train_loader)

# For each entry in eval data set, check models prediction against actual label.
def evaluate(model: nn.Module, eval_loader: DataLoader) -> tuple[float, list[int], list[int]]:

    # Switch to eval mode to disable dropout
    model.eval()

    # Stop tracking gradients since we won't use gradients from evaluation operations.
    with torch.no_grad():
        predictions = []
        gold_labels = []

        for inputs, labels in eval_loader:

            # Move batch to the same device as the model
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Get model scores
            logits = model(inputs)

            # Get highest score prediction
            prediction = logits.argmax(dim=1)

            # Append batch predictions and gold labels to lists
            predictions.extend(prediction.tolist())
            gold_labels.extend(labels.tolist())

    correct = 0

    # Count correct predictions
    for i in range(len(predictions)):
        if predictions[i] == gold_labels[i]:
            correct += 1

    accuracy = correct / len(predictions)

    return accuracy, predictions, gold_labels

# Train a model for training_epochs, validating after every epoch, and keep the weights from the epoch with the best validation accuracy to prevent overfitting.
# Shared by the CNN and RNN so both models are trained the exact same way.
# model_name, condition, and training_domain only label the rows written to the training logs.
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, training_epochs: int, model_name: str = "", condition: str = "", training_domain: str = "all") -> nn.Module:

    # Move all weights to the GPU if available. Must happen before creating the optimizer.
    model.to(DEVICE)

    print(f"[train_model] Training on device: {DEVICE}")

    # Number of weights the optimizer updates
    num_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    print(f"[train_model] Trainable parameters: {num_parameters}")

    run_labels = [model_name, condition, training_domain]

    training_start_time = time.perf_counter()

    #  Soft max loss function (-log(correct class probability))
    loss_function = nn.CrossEntropyLoss()

    # Optimizer used for updated weights using Adam to pick learning rate for each weight.
    optimizer = torch.optim.Adam(params=model.parameters())

    # Track best accuracy to use weights with best validation performance preventing overfitting. Starts below 0 so the first epoch is always kept.
    best_epoch_accuracy = -1

    best_weights = None

    best_epoch = None

    # Train, then evaluate against validate data set
    for i in range(training_epochs):
        epoch_num = i + 1
        print(f"[train_model] Training epoch {epoch_num}")
        epoch_start_time = time.perf_counter()
        epoch_loss = train_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer, loss_function=loss_function)
        print(f"[train_model] Finished training epoch {epoch_num}, loss: {epoch_loss}")
        print(f"[train_model] Validating after epoch {epoch_num}")
        accuracy, _, _ = evaluate(model=model, eval_loader=val_loader)
        epoch_seconds = time.perf_counter() - epoch_start_time

        append_training_log(TRAINING_HISTORY_PATH, TRAINING_HISTORY_HEADER, run_labels + [epoch_num, epoch_loss, accuracy, epoch_seconds])

        # Update best epoch weights
        if accuracy > best_epoch_accuracy:
            best_epoch_accuracy = accuracy
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch_num

        print(f"[train_model] Validated after epoch {epoch_num}, accuracy: {accuracy}")

    # Load weights with best validation performance
    model.load_state_dict(best_weights)

    print(f"[train_model] Finished training for {training_epochs} epochs")
    print(f"[train_model] best epoch: {best_epoch}, best validation accuracy: {best_epoch_accuracy}")

    total_seconds = time.perf_counter() - training_start_time

    print(f"[train_model] Total training time: {total_seconds:.1f} seconds")

    append_training_log(TRAINING_SUMMARY_PATH, TRAINING_SUMMARY_HEADER, run_labels + [num_parameters, training_epochs, best_epoch, best_epoch_accuracy, total_seconds, str(DEVICE)])

    return model
