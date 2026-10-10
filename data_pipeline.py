import ftfy
import pandas as pd

# Sources to keep in the dataset. Other sources like 'Human Written' will be filtered out.
KEEP_SOURCES = ["ChatGPT", "Claude", "Mistral AI", "Copilot", "Qwen"]

# Question ids where at least one kept source has an answer that belongs to a different question (shifted or duplicated answers in the raw dataset).
# Found by comparing each answer to the other models' answers for every question in the same subject. Dropped for every source so all sources keep the same questions.
MISMATCHED_QUESTION_IDS = {
    202, 204, 205, 206, 207, 208, 281, 282, 283, 284, 286, 287, 288, 289, 290, 291,
    292, 293, 294, 295, 296, 297, 298, 299, 367, 495, 680, 681, 728, 848, 864, 879,
    901, 961, 962, 963, 964, 965, 966, 1003, 1005, 1021, 1022, 1023, 1024, 1027,
    1060, 1263, 1289, 1291, 1409, 1422, 1495, 1498, 1499, 1557, 1574, 1576, 1704,
    1705, 1708, 1789, 1879, 1886, 1887, 1888, 1896, 1947, 1948, 1949, 1976, 1977,
    1978, 1979, 1980, 1982, 1984, 1985, 1986, 1987, 1989, 1990, 1991, 1992, 1993,
    1994, 1995, 1996, 1997, 1998, 2183, 2185, 2186, 2189, 2191, 2192, 2193, 2194,
    2195, 2196, 2198, 2249, 2250, 2303, 2346, 2347, 2358, 2372, 2499, 2699, 2718,
    2748, 2750, 2807, 2811, 2855, 2856, 2857, 2880, 2881, 2882, 2883, 2884, 2885,
    2886, 2887, 2888, 2889, 2890, 2891, 2893, 2894, 2895, 2896, 2897, 2898, 2899,
    2987,
}

# Question ids where at least one kept source has a cut-off answer (ends mid-sentence). Dropped for every source like the mismatched questions.
CUT_OFF_QUESTION_IDS = {650, 713, 715, 1052, 1759}

# In the raw dataset, Claude's last Gender Studies question ("How can men and women collaborate for gender justice?") is placed at the end of the Agriculture block (position 2899 instead of 2799), shifting Claude's rows 2799-2898 back by one.
CLAUDE_MISPLACED_POSITION = 2899
CLAUDE_CORRECT_POSITION = 2799

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

# Move Claude's misplaced Gender Studies row back to its correct position so Claude's rows line up with the other sources.
def fix_claude_row_order(df: pd.DataFrame) -> pd.DataFrame:
    claude_rows = df[df["Source"] == "Claude"]
    order = list(range(len(claude_rows)))
    misplaced = order.pop(CLAUDE_MISPLACED_POSITION)
    order.insert(CLAUDE_CORRECT_POSITION, misplaced)
    other_rows = df[df["Source"] != "Claude"]
    return pd.concat([other_rows, claude_rows.iloc[order]])

# Number questions from each source so that the same question from a different source has the same question_id. This is used to map the same question from different sources to the same subject and domain.
def number_questions_within_each_model(df: pd.DataFrame) -> pd.DataFrame:
    numbered_models = []
    for model in KEEP_SOURCES:
        model_rows = df[df["Source"] == model].copy()
        model_rows["question_id"] = range(len(model_rows))
        numbered_models.append(model_rows)
    return pd.concat(numbered_models, ignore_index=True)

# Drop questions with mismatched answers from every source and print how many rows were removed.
def drop_mismatched_questions(df: pd.DataFrame) -> pd.DataFrame:
    is_mismatched = df["question_id"].isin(MISMATCHED_QUESTION_IDS)
    kept = df[~is_mismatched].copy()
    print(f"[drop_mismatched_questions] rows before: {len(df)}, dropped: {is_mismatched.sum()}, rows after: {len(kept)}")
    print(f"[drop_mismatched_questions] rows per source after drop: {kept['Source'].value_counts().to_dict()}")
    return kept

# Drop questions with cut-off answers from every source and print how many rows were removed.
def drop_cut_off_questions(df: pd.DataFrame) -> pd.DataFrame:
    is_cut_off = df["question_id"].isin(CUT_OFF_QUESTION_IDS)
    kept = df[~is_cut_off].copy()
    print(f"[drop_cut_off_questions] rows before: {len(df)}, dropped: {is_cut_off.sum()}, rows after: {len(kept)}")
    print(f"[drop_cut_off_questions] rows per source after drop: {kept['Source'].value_counts().to_dict()}")
    return kept

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

    df = fix_claude_row_order(df)

    df = number_questions_within_each_model(df)

    df = use_one_wording_per_question(df)

    df = drop_mismatched_questions(df)

    df = drop_cut_off_questions(df)

    df["subject"] = df["question_id"].map(subject_of_question)
    df["domain"] = df["subject"].map(SUBJECT_DOMAINS)

    df.to_csv(out_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    main(IN_PATH, OUT_PATH)