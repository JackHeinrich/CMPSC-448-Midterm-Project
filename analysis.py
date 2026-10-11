import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score

def create_confusion_matrix(predictions: list[int], gold_labels: list[int], label_names: list[str], title: str) -> None:
    #Map names to an int
    label_ids = list(range(len(label_names)))

    #calculate confusion % for each cell
    row_fractions = confusion_matrix(y_true=gold_labels, y_pred=predictions, labels=label_ids, normalize="true")

    # Create image
    figure, axes = plt.subplots(figsize=(7, 6))

    # Package table with display names for axis
    display = ConfusionMatrixDisplay(confusion_matrix=row_fractions, display_labels=label_names)

    # Draw table
    display.plot(ax=axes, cmap="Blues", values_format=".0%", colorbar=False)
    
    axes.set_title(title)
    axes.set_xlabel("Predicted model")
    axes.set_ylabel("True model")

    # Rotate x tick labels by 30 deg si they dont overlap
    plt.setp(axes.get_xticklabels(), rotation=30, ha="right")
    
    figure.tight_layout()

    # Save to results
    figure.savefig(f"results/{title}.png")

    plt.close(figure)

# Fraction of each LLM's responses that were correctly identified (recall), as {label name: recall}.
# These are the same numbers as the diagonal of the confusion matrix.
def per_label_recall(predictions: list[int], gold_labels: list[int], label_names: list[str]) -> dict[str, float]:
    recalls = recall_score(y_true=gold_labels, y_pred=predictions, labels=list(range(len(label_names))), average=None, zero_division=0)
    return {label_name: float(recall) for label_name, recall in zip(label_names, recalls)}

# Draw research question 3's grid of accuracies, with training domains as rows and testing domains as columns.
# The diagonal is in-domain accuracy. Reading down a column compares in-domain and cross-domain accuracy on the same test questions.
def create_domain_accuracy_grid(accuracies: dict[tuple[str, str], float], domains: list[str], title: str) -> None:

    # grid[row][column] = accuracy of the model trained on domains[row] and tested on domains[column]
    grid = np.array([[accuracies[(training_domain, testing_domain)] for testing_domain in domains] for training_domain in domains])

    # Create image
    figure, axes = plt.subplots(figsize=(7, 6))

    # Draw table, scaled from 0 to 100% so grids for different models are comparable
    axes.imshow(grid, cmap="Blues", vmin=0, vmax=1)

    # Write the accuracy in each cell, using white text on dark cells so it stays readable
    for row in range(len(domains)):
        for column in range(len(domains)):
            text_color = "white" if grid[row, column] > 0.6 else "black"
            axes.text(column, row, f"{grid[row, column]:.0%}", ha="center", va="center", color=text_color)

    axes.set_xticks(range(len(domains)), labels=domains)
    axes.set_yticks(range(len(domains)), labels=domains)

    axes.set_title(title)
    axes.set_xlabel("Testing domain")
    axes.set_ylabel("Training domain")

    figure.tight_layout()

    # Save to results
    figure.savefig(f"results/{title}.png")

    plt.close(figure)

# Draw training loss and validation accuracy per epoch for one model's research question 1 and 2 runs (trained on all domains), with one line per condition.
def create_training_curves(history_path: str, model_name: str, title: str) -> None:

    history = pd.read_csv(history_path)

    # Only the runs trained on all domains, which are the research question 1 and 2 runs
    history = history[(history["model"] == model_name) & (history["training domain"] == "all")]

    # Create image with loss on the left and validation accuracy on the right
    figure, (loss_axes, accuracy_axes) = plt.subplots(1, 2, figsize=(11, 4.5))

    for condition, runs in history.groupby("condition", sort=False):
        loss_axes.plot(runs["epoch"], runs["training loss"], marker="o", label=condition)
        accuracy_axes.plot(runs["epoch"], runs["validation accuracy"], marker="o", label=condition)

    loss_axes.set_title("Training loss")
    loss_axes.set_xlabel("Epoch")
    loss_axes.set_ylabel("Average cross-entropy loss")

    accuracy_axes.set_title("Validation accuracy")
    accuracy_axes.set_xlabel("Epoch")
    accuracy_axes.set_ylabel("Accuracy")
    accuracy_axes.set_ylim(0, 1)

    for axes in (loss_axes, accuracy_axes):
        # Only whole epochs on the x axis
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))
        axes.legend(title="Condition")
        axes.grid(alpha=0.3)

    figure.suptitle(title)

    figure.tight_layout()

    # Save to results
    figure.savefig(f"results/{title}.png")

    plt.close(figure)

def main() -> None:
    create_confusion_matrix([0,0,1,2,3], [1,0,0,0,4], ["john", "mark", "ham", "rob", "pete"], "test")

if __name__ == "__main__":
    main()