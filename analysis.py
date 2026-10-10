import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

def main() -> None:
    create_confusion_matrix([0,0,1,2,3], [1,0,0,0,4], ["john", "mark", "ham", "rob", "pete"], "test")

if __name__ == "__main__":
    main()