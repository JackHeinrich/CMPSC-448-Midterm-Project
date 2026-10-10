import csv
from analysis import create_confusion_matrix, create_domain_accuracy_grid, per_label_recall
from cnn import evaluate_cnn_on_condition, evaluate_cnn_across_domains
from rnn import evaluate_rnn_on_condition, evaluate_rnn_across_domains

from text_pipeline import DOMAINS

TRAINING_EPOCHS = 10

RESULTS_PATH = "results/accuracies.csv"

# Recall of each LLM for every research question 3 (training domain, testing domain) pair. Used to compare which LLMs stay identifiable across domains.
RECALL_RESULTS_PATH = "results/rq3_recall.csv"

# Research question 3 produces a confusion matrix for every (training domain, testing domain) pair, 18 in total. Off by default since RECALL_RESULTS_PATH summarizes them.
SAVE_DOMAIN_CONFUSION_MATRICES = False

# Conditions for research question 2 and how they are named in figure titles.
CONDITION_TITLES = {
    "outputs": "Outputs Only",
    "inputs": "Inputs Only",
    "both": "Both"
}

# Each neural network and its functions for (research questions 1 and 2, research question 3).
MODELS = {
    "cnn": (evaluate_cnn_on_condition, evaluate_cnn_across_domains),
    "rnn": (evaluate_rnn_on_condition, evaluate_rnn_across_domains)
}

# Write every result so far to RESULTS_PATH. Called after each result so a crash part way through doesn't lose earlier results.
def write_results(rows: list[tuple]) -> None:
    with open(RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "condition", "training domain", "testing domain", "accuracy"])
        writer.writerows(rows)

# Write every research question 3 recall row so far to RECALL_RESULTS_PATH, with one recall column per LLM.
def write_recall_results(rows: list[tuple], label_names: list[str]) -> None:
    with open(RECALL_RESULTS_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "training domain", "testing domain"] + [f"{label_name} recall" for label_name in label_names])
        writer.writerows(rows)

# Research questions

def main() -> None:

    # Rows of (model, condition, training domain, testing domain, accuracy)
    rows = []

    # Rows of (model, training domain, testing domain, recall of each LLM)
    recall_rows = []

    # Questions 1 and 2: train each model on every condition using all domains.

    for model_name, (evaluate_on_condition, _) in MODELS.items():
        for condition, condition_title in CONDITION_TITLES.items():

            accuracy, predictions, gold, label_names = evaluate_on_condition(condition, TRAINING_EPOCHS)

            create_confusion_matrix(predictions, gold, label_names, f"{model_name.upper()} {condition_title} Confusion Matrix")

            rows.append((model_name, condition, "all", "all", accuracy))
            write_results(rows)

    # Question 3: train each model once per domain on outputs only, then test it on every domain's test questions.

    for model_name, (_, evaluate_across_domains) in MODELS.items():

        # {(training domain, testing domain): accuracy}
        domain_accuracies = {}

        for training_domain in DOMAINS:

            results, label_names = evaluate_across_domains("outputs", TRAINING_EPOCHS, training_domain)

            for testing_domain, (accuracy, predictions, gold) in results.items():

                domain_accuracies[(training_domain, testing_domain)] = accuracy

                if SAVE_DOMAIN_CONFUSION_MATRICES:
                    create_confusion_matrix(predictions, gold, label_names, f"{model_name.upper()} Trained on {training_domain} Tested on {testing_domain} Confusion Matrix")

                rows.append((model_name, "outputs", training_domain, testing_domain, accuracy))
                write_results(rows)

                recalls = per_label_recall(predictions, gold, label_names)
                recall_rows.append((model_name, training_domain, testing_domain, *[recalls[label_name] for label_name in label_names]))
                write_recall_results(recall_rows, label_names)

        create_domain_accuracy_grid(domain_accuracies, DOMAINS, f"{model_name.upper()} Cross Domain Accuracy")

    # Print every result together at the end
    print("[Main] Results (model, condition, training domain, testing domain, accuracy):")
    for row in rows:
        print(f"[Main] {row}")

if __name__ == "__main__":
    main()
