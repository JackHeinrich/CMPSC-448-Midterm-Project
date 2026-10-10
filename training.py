import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Do one pass through the training data set and update weights
def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: nn.CrossEntropyLoss) -> float:

    # Training mode turns dropout on
    model.train()

    total_loss = 0

    # Loop over every batch
    for inputs, labels in train_loader:

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