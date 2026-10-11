import numpy as np
import pandas as pd

IN_PATH = "data/TXD-22 Filtered and Labeled.csv"

# The cut of the dataset used for training, validation, and testing. (1 - TRAIN_PERCENT - VAL_PERCENT) is the test percent.
TRAIN_PERCENT, VAL_PERCENT = 0.7, 0.1

# Seed for rng, can be anything
SEED = 1

# Assigns each question to the train, validation, or test split.
def assign_splits(df: pd.DataFrame) -> dict[int, str]:
    print(f"[assign_splits] Assigning splits for {len(df)} rows")
    questions = df[["question_id", "subject"]].drop_duplicates()
    
    rng = np.random.RandomState(SEED)
    
    split_of_question = {}

    for subject, group in questions.groupby("subject"):

        shuffled_ids = rng.permutation(group["question_id"].to_numpy())

        amount_for_train = round(len(shuffled_ids) * TRAIN_PERCENT)

        amount_for_val = round(len(shuffled_ids) * VAL_PERCENT)

        for position, question_id in enumerate(shuffled_ids):
            if position < amount_for_train:
                split_of_question[question_id] = "train"
            elif position < amount_for_train + amount_for_val:
                split_of_question[question_id] = "val"
            else:
                split_of_question[question_id] = "test"

    print("[assign_splits] Finished assigning splits.")

    return split_of_question

# Load the train, validation, and test splits and return them as a tuple.
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"[load_splits] Loading splits from {IN_PATH}")
    df = pd.read_csv(IN_PATH)
    df["split"] = df["question_id"].map(assign_splits(df))

    print("[load_splits] finished loading splits.")

    return (df[df["split"] == "train"],
            df[df["split"] == "val"],
            df[df["split"] == "test"])

# Check that it works and the splits seem reasonable.
if __name__ == "__main__":
    splits = load_splits()
    print(f"Train contains {len(splits[0])} rows, Val contains {len(splits[1])} rows, Test contains {len(splits[2])} rows")