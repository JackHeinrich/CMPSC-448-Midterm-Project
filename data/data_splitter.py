import numpy as np
import pandas as pd

IN_PATH = "data\\TXD-22 Filtered and Labeled.csv"
TRAIN_PERCENT, VAL_PERCENT = 0.7, 0.1
SEED = 1

def assign_splits(df: pd.DataFrame) -> dict[int, str]:
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

    return split_of_question

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(IN_PATH)
    df["split"] = df["question_id"].map(assign_splits(df))
    return (df[df["split"] == "train"],
            df[df["split"] == "val"],
            df[df["split"] == "test"])

if __name__ == "__main__":
    splits = load_splits()
    print(f"Train contains {len(splits[0])} rows, Val contains {len(splits[1])} rows, Test contains {len(splits[2])} rows")