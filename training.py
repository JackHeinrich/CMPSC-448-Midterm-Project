import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Adapted from Chris Tran's "A Complete Guide to CNN for Sentence Classification with PyTorch"
def get_device() -> torch.device:
    if torch.cuda.is_available():       
        device = torch.device("cuda")
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device

# Get device that a model is on
def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device

# Do one pass through the training data set and update weights
def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, loss_function: nn.CrossEntropyLoss) -> float:

    # Training mode turns dropout on
    model.train()

    device = get_model_device(model)

    total_loss = 0

    # Loop over every batch
    for inputs, labels in train_loader:

        # Move inputs and labels to model device (hopefully a GPU)
        inputs = inputs.to(device)
        labels=  labels.to(device)

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

            inputs = inputs.to(device)
            labels=  labels.to(device)

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