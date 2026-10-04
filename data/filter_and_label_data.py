import ftfy
import pandas as pd

KEEP_SOURCES = ["ChatGPT", "Claude", "Gemini", "Llama", "Qwen"]

QUESTIONS_PER_SOURCE = 3000
SUBJECT_SIZE = 100

STEM = "STEM"
SOCIAL_SCIENCE = "Social Science"
HUMANITIES = "Humanities"

IN_PATH = "data\\TXD-22 A Large-Scale Benchmark Dataset.csv"
OUT_PATH = "data\\TXD-22 Filtered and Labeled.csv"

SUBJECT_DOMAINS = {
    "Computer Science":              STEM,
    "Electrical Engineering":        STEM,
    "Economics":                     SOCIAL_SCIENCE,
    "Sociology":                     SOCIAL_SCIENCE,
    "Business & Management":         SOCIAL_SCIENCE,
    "Pharmacy":                      STEM,
    "English & Communication":       HUMANITIES,
    "Law":                           SOCIAL_SCIENCE,
    "Botany":                        STEM,
    "Criminology":                   SOCIAL_SCIENCE,
    "Disaster Science":              SOCIAL_SCIENCE,
    "Geography & Environment":       SOCIAL_SCIENCE,
    "Graphic Design":                HUMANITIES,
    "History":                       HUMANITIES,
    "Information & Library Science": HUMANITIES,
    "International Relations":       SOCIAL_SCIENCE,
    "Mass Communication":            HUMANITIES,
    "Nuclear Engineering":           STEM,
    "Philosophy":                    HUMANITIES,
    "Physics":                       STEM,
    "Political Science":             SOCIAL_SCIENCE,
    "Psychology":                    SOCIAL_SCIENCE,
    "Robotics":                      STEM,
    "Statistics":                    STEM,
    "Film & Television":             HUMANITIES,
    "Theatre":                       HUMANITIES,
    "Tourism":                       SOCIAL_SCIENCE,
    "Gender Studies":                SOCIAL_SCIENCE,
    "Agriculture":                   STEM,
    "Zoology":                       STEM,
}

SUBJECTS = list(SUBJECT_DOMAINS)

def keep_only_selected_models(df: pd.DataFrame) -> pd.DataFrame:
    is_selected_model = df["Source"].isin(KEEP_SOURCES)
    return df[is_selected_model].copy()

def subject_of_question(question_id: int) -> str:
    subject_index = question_id // SUBJECT_SIZE
    return SUBJECTS[subject_index]

def fix_garbled_text(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["Question", "Answer"]:
        df[column] = df[column].astype(str).map(ftfy.fix_text).str.strip()
    return df

def number_questions_within_each_model(df: pd.DataFrame) -> pd.DataFrame:
    numbered_models = []
    for model in KEEP_SOURCES:
        model_rows = df[df["Source"] == model].copy()
        model_rows["question_id"] = range(len(model_rows))
        numbered_models.append(model_rows)
    return pd.concat(numbered_models, ignore_index=True)

def main(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path, encoding="latin-1")

    df = keep_only_selected_models(df)
    df = fix_garbled_text(df)

    df = number_questions_within_each_model(df)

    df["subject"] = df["question_id"].map(subject_of_question)
    df["domain"] = df["subject"].map(SUBJECT_DOMAINS)

    df.to_csv(out_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main(IN_PATH, OUT_PATH)