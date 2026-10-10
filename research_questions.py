import csv
from analysis import create_confusion_matrix
from cnn import evaluate_cnn_on_condition

from data_pipeline import STEM, HUMANITIES

TRAINING_EPOCHS = 10

# Research questions

def main() -> None:
    # Questions 1 and 2

    cnn_outputs_accuracy, cnn_outputs_predictions, cnn_outputs_gold, cnn_label_names = evaluate_cnn_on_condition("outputs", TRAINING_EPOCHS)

    create_confusion_matrix(cnn_outputs_predictions, cnn_outputs_gold, cnn_label_names, "CNN Outputs Only Confusion Matrix")
    
    cnn_inputs_accuracy, cnn_inputs_predictions, cnn_inputs_gold, _ = evaluate_cnn_on_condition("inputs", TRAINING_EPOCHS)

    create_confusion_matrix(cnn_inputs_predictions, cnn_inputs_gold, cnn_label_names, "CNN Inputs Only Confusion Matrix")

    cnn_both_accuracy, cnn_both_predictions, cnn_both_gold, _ = evaluate_cnn_on_condition("both", TRAINING_EPOCHS)

    create_confusion_matrix(cnn_both_predictions, cnn_both_gold, cnn_label_names, "CNN Both Confusion Matrix")

    print(f"[Main] Research question 1 result: CNN Accuracy: {cnn_outputs_accuracy}, RNN Accuracy: ")

    print(f"[Main] Research question 2 result:\n\nCNN: outputs only accuracy: {cnn_outputs_accuracy}, inputs only accuracy: {cnn_inputs_accuracy}, both accuracy: {cnn_both_accuracy}.\n\nRNN: outputs only accuracy: REPLACE ME, inputs only accuracy: REPLACE ME, both accuracy: REPLACE ME")

    # Questions 3, 4, 5

    cnn_outputs_in_domain_accuracy, cnn_outputs_in_domain_predictions, cnn_outputs_in_domain_gold, cnn_label_names = evaluate_cnn_on_condition("outputs", TRAINING_EPOCHS, STEM, STEM)

    cnn_outputs_cross_domain_accuracy, cnn_outputs_cross_domain_predictions, cnn_outputs_cross_domain_gold, _ = evaluate_cnn_on_condition("outputs", TRAINING_EPOCHS, STEM, HUMANITIES)

    create_confusion_matrix(cnn_outputs_cross_domain_predictions, cnn_outputs_cross_domain_gold, cnn_label_names, "CNN Outputs Cross Domain Confusion Matrix")

    create_confusion_matrix(cnn_outputs_in_domain_predictions, cnn_outputs_in_domain_gold, cnn_label_names, "CNN Outputs In Domain Confusion Matrix")


    print(f"[Main] Research question 3 result: \n\nCNN Outputs Cross Domain Accuracy: {cnn_outputs_cross_domain_accuracy}, CNN Outputs In Domain Accuracy: {cnn_outputs_in_domain_accuracy}")

    rows = [
        ("cnn", "outputs", "all", "all", cnn_outputs_accuracy),
        ("cnn", "inputs", "all", "all", cnn_inputs_accuracy),
        ("cnn", "both", "all", "all", cnn_both_accuracy),
        ("cnn", "outputs", STEM, STEM, cnn_outputs_in_domain_accuracy),
        ("cnn", "outputs", STEM, HUMANITIES, cnn_outputs_cross_domain_accuracy)
    ]

    with open("results/accuracies.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "condition", "training domain", "testing domain", "accuracy"])
        writer.writerows(rows)

if __name__ == "__main__":
    main()