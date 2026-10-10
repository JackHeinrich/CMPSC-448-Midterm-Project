import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Train on the GPU if one with CUDA is available, otherwise fall back to the CPU. Models must be moved here with model.to(DEVICE) before training.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Do one pass through the training data set and update weights
def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: nn.CrossEntropyLoss) -> float:

    # Training mode turns dropout on
    model.train()

    device = get_model_device(model)

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

    device = get_model_device(model)

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