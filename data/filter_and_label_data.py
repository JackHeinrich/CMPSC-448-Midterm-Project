import ftfy
import pandas as pd

# Sources to keep in the dataset. Other sources like 'Human Written' will be filtered out.
KEEP_SOURCES = ["ChatGPT", "Claude", "Gemini", "Llama", "Qwen"]

# Subject size used to determine the subject of a question as all entries for a source have the same order of topics with each topic having 100 entries 
SUBJECT_SIZE = 100

# Domain categories for the extra credit research question 3.
STEM = "STEM"
SOCIAL_SCIENCE = "Social Science"
HUMANITIES = "Humanities"

IN_PATH = "data\\TXD-22 A Large-Scale Benchmark Dataset.csv"
OUT_PATH = "data\\TXD-22 Filtered and Labeled.csv"

# Prompt topics in order mapped to their domain for extra credit question 3.
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

# Flat list of subjects to get subject of a prompt from it's index.
SUBJECTS = list(SUBJECT_DOMAINS)

# Filter out unwanted sources
def keep_only_selected_models(df: pd.DataFrame) -> pd.DataFrame:
    is_selected_model = df["Source"].isin(KEEP_SOURCES)
    return df[is_selected_model].copy()

# Assign subject of question by its index
def subject_of_question(question_id: int) -> str:
    subject_index = question_id // SUBJECT_SIZE
    return SUBJECTS[subject_index]

# Use ftfy to fix garbled text from needing to open the dataset with latin-1 encoding instead of utf-8. Also replace Windows line endings with Unix line endings and strip whitespace.
def fix_garbled_text(df: pd.DataFrame) -> pd.DataFrame:
    for column in ["Question", "Answer"]:
        df[column] = (
            df[column]
            .astype(str)
            .map(ftfy.fix_text)
            .str.replace("\r\n", "\n")
            .str.strip()
        )
    return df

# Number questions from each source so that the same question from a different source has the same question_id. This is used to map the same question from different sources to the same subject and domain.
def number_questions_within_each_model(df: pd.DataFrame) -> pd.DataFrame:
    numbered_models = []
    for model in KEEP_SOURCES:
        model_rows = df[df["Source"] == model].copy()
        model_rows["question_id"] = range(len(model_rows))
        numbered_models.append(model_rows)
    return pd.concat(numbered_models, ignore_index=True)

# This unifies the prompts from different sources as the original dataset sometimes has small differences between the prompts asked to different llm models.
def use_one_wording_per_question(df: pd.DataFrame) -> pd.DataFrame:
    chatgpt_rows = df[df["Source"] == "ChatGPT"]
    chatgpt_wording = chatgpt_rows.set_index("question_id")["Question"]
    df["Question"] = df["question_id"].map(chatgpt_wording)
    return df

# Main data processing function
def main(in_path: str, out_path: str) -> None:
    df = pd.read_csv(in_path, encoding="latin-1")

    df = keep_only_selected_models(df)

    df = fix_garbled_text(df)

    df = number_questions_within_each_model(df)

    df = use_one_wording_per_question(df)

    df["subject"] = df["question_id"].map(subject_of_question)
    df["domain"] = df["subject"].map(SUBJECT_DOMAINS)

    df.to_csv(out_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main(IN_PATH, OUT_PATH)