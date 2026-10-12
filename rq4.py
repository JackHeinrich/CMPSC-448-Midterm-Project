import math
import re
from collections import Counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from analysis import per_label_recall
from text_pipeline import get_data_loaders, tokenize
from training import evaluate, train_model

# Research question 4: what characteristics distinguish the LLMs?
# Part 1 describes how the LLMs' answers differ (no training).
# Part 2 retrains a model with one signal removed at a time and measures how much accuracy drops.

DATA_PATH = "data/TXD-22 Filtered and Labeled.csv"

STYLE_STATISTICS_PATH = "results/rq4_style_statistics.csv"

CHARACTERISTIC_WORDS_PATH = "results/rq4_characteristic_words.csv"

# Number of characteristic words to list for each LLM
CHARACTERISTIC_WORDS_PER_LLM = 15

# Only consider words used at least this many times by an LLM as characteristic, so rare words don't dominate.
CHARACTERISTIC_WORD_MIN_COUNT = 20

# Window size for lexical diversity. Plain distinct words / total words falls as texts get longer, so it is averaged over fixed size windows instead.
LEXICAL_DIVERSITY_WINDOW = 50

# Length of the word sequence checked for repetition. An answer that uses the same sequence of this many words twice is counted as repeating itself.
REPEATED_PHRASE_WORDS = 8

# Answers are cut to this many tokens to remove the length signal. Over 99% of answers are at least this long, so all answers become the same length.
EQUAL_LENGTH_TOKENS = 100

# Placeholder tokens that replace hidden words, so the hidden words still take up their position in the sequence.
CONTENT_PLACEHOLDER = "<content>"
FUNCTION_PLACEHOLDER = "<function>"

# Function words: grammatical words that carry little topic meaning, such as articles, pronouns, prepositions, conjunctions, auxiliary verbs, and common adverbs.
# Includes the pieces our tokenizer splits contractions into (don't -> don ' t).
FUNCTION_WORDS = set("""
a an the this that these those such
i me my mine we us our ours you your yours he him his she her hers it its they them their theirs
who whom whose which what whatever whichever whoever
myself yourself himself herself itself ourselves yourselves themselves
and or but nor so yet for if then else than because although though while whereas unless until since whether
as also either neither both each every all any some no not none only own same other another more most much many few several
of to in on at by with without from into onto upon over under above below between among through throughout during before after
about against along around across behind beyond toward towards within via per like unlike despite
is am are was were be been being have has had having do does did doing done
can could may might must shall should will would ought
there here where when why how wherever whenever
very too quite rather just even still already again ever never always often sometimes however therefore thus hence moreover furthermore
indeed instead meanwhile otherwise nevertheless nonetheless additionally consequently accordingly
s t d ll re ve m don doesn didn isn aren wasn weren hasn haven hadn won wouldn shouldn couldn can cannot
""".split())

# Matches one word in the original text
WORD_PATTERN = re.compile(r"\b\w+\b")

# Splits a text into sentences after ., !, or ?
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

# --- Part 1: describe how the LLMs differ ---

# Average fraction of distinct words in every window of LEXICAL_DIVERSITY_WINDOW consecutive words (moving average type-token ratio).
# Unlike distinct words / total words, this doesn't depend on how long the answer is.
def moving_average_type_token_ratio(words: list[str]) -> float:

    window = LEXICAL_DIVERSITY_WINDOW

    # Answers shorter than the window fall back to the plain ratio
    if len(words) <= window:
        return len(set(words)) / max(len(words), 1)

    # Slide the window one word at a time, updating the word counts instead of recounting every window
    counts = Counter(words[:window])
    ratios = [len(counts) / window]

    for i in range(window, len(words)):
        leaving_word = words[i - window]
        counts[leaving_word] -= 1
        if counts[leaving_word] == 0:
            del counts[leaving_word]
        counts[words[i]] += 1
        ratios.append(len(counts) / window)

    return sum(ratios) / len(ratios)

# Whether any sequence of REPEATED_PHRASE_WORDS words appears more than once in the answer.
def has_repeated_phrase(words: list[str]) -> bool:
    phrases = [tuple(words[i:i + REPEATED_PHRASE_WORDS]) for i in range(len(words) - REPEATED_PHRASE_WORDS + 1)]
    return len(set(phrases)) < len(phrases)

# Measure the style of one answer. Each value is averaged per LLM in compute_style_statistics.
def measure_answer(text: str) -> dict[str, float]:

    words = WORD_PATTERN.findall(text.lower())
    num_words = max(len(words), 1)

    sentences = [sentence for sentence in SENTENCE_SPLIT_PATTERN.split(text.strip()) if sentence]

    # Punctuation counts are scaled to per 100 words so longer answers don't count more
    def per_100_words(count: int) -> float:
        return 100 * count / num_words

    return {
        # Length (structural)
        "words per answer": len(words),
        "tokens per answer": len(tokenize(text)),
        "sentences per answer": len(sentences),
        "words per sentence": len(words) / max(len(sentences), 1),
        # Formatting (structural)
        "has line break": float("\n" in text),
        "has list": float(re.search(r"(?m)^\s*(?:[-*•]|\d+[.)])\s", text) is not None),
        "has bold": float("**" in text),
        # Punctuation (structural)
        "commas per 100 words": per_100_words(text.count(",")),
        "colons per 100 words": per_100_words(text.count(":")),
        "semicolons per 100 words": per_100_words(text.count(";")),
        "dashes per 100 words": per_100_words(text.count("—") + text.count("–") + text.count(" - ")),
        "parentheses per 100 words": per_100_words(text.count("(")),
        "has dash": float(any(dash in text for dash in ("—", "–", " - "))),
        # Vocabulary (linguistic)
        "lexical diversity": moving_average_type_token_ratio(words),
        "average word length": sum(len(word) for word in words) / num_words,
        "function word share": sum(word in FUNCTION_WORDS for word in words) / num_words,
        "has repeated 8 word phrase": float(has_repeated_phrase(words)),
    }

# Average every style measure for each LLM. Rows are measures and columns are LLMs, which reads more easily in a report.
def compute_style_statistics(df: pd.DataFrame) -> pd.DataFrame:
    measures = pd.DataFrame([measure_answer(text) for text in df["Answer"]])
    measures["Source"] = df["Source"].to_numpy()
    return measures.groupby("Source").mean().T

# Words each LLM uses much more often than the other LLMs combined, ranked by a z-score of the log odds ratio
# (Monroe, Colaresi & Quinn, 2008, "Fightin' Words"). Every LLM answered the same questions, so these reflect word choice rather than topic.
def find_characteristic_words(df: pd.DataFrame) -> pd.DataFrame:

    # Count words for each LLM. Only alphabetic words, so numbers and punctuation are left out.
    word_counts = {source: Counter() for source in sorted(df["Source"].unique())}
    for text, source in zip(df["Answer"], df["Source"]):
        word_counts[source].update(word for word in tokenize(text) if word.isalpha())

    total_counts = Counter()
    for counts in word_counts.values():
        total_counts.update(counts)

    # Small prior added to every count so words that one LLM never uses don't divide by zero
    prior = 0.01
    vocab_size = len(total_counts)

    rows = []

    for source, counts in word_counts.items():
        source_total = sum(counts.values())
        others_total = sum(total_counts.values()) - source_total

        scores = []
        for word, count in counts.items():
            if count < CHARACTERISTIC_WORD_MIN_COUNT:
                continue
            others_count = total_counts[word] - count
            log_odds_source = math.log((count + prior) / (source_total + prior * vocab_size - count - prior))
            log_odds_others = math.log((others_count + prior) / (others_total + prior * vocab_size - others_count - prior))
            variance = 1 / (count + prior) + 1 / (others_count + prior)
            scores.append(((log_odds_source - log_odds_others) / math.sqrt(variance), word, count, others_count))

        scores.sort(reverse=True)

        for rank, (z_score, word, count, others_count) in enumerate(scores[:CHARACTERISTIC_WORDS_PER_LLM], start=1):
            # Uses per 10,000 words, so LLMs with longer answers are compared fairly
            rows.append({
                "LLM": source,
                "rank": rank,
                "word": word,
                "z score": round(z_score, 2),
                "uses per 10k words (this LLM)": round(10_000 * count / source_total, 2),
                "uses per 10k words (other LLMs)": round(10_000 * others_count / others_total, 2),
            })

    return pd.DataFrame(rows)

# Box plot of answer length for each LLM.
def create_answer_length_plot(df: pd.DataFrame, title: str) -> None:

    sources = sorted(df["Source"].unique())
    lengths = [df[df["Source"] == source]["Answer"].map(lambda text: len(WORD_PATTERN.findall(text))) for source in sources]

    figure, axes = plt.subplots(figsize=(7, 5))
    axes.boxplot(lengths, showfliers=False)

    # Name the boxes with set_xticklabels rather than boxplot's label argument, which was renamed in matplotlib 3.9 (labels -> tick_labels) and fails on other versions
    axes.set_xticks(range(1, len(sources) + 1))
    axes.set_xticklabels(sources)

    axes.set_title(title)
    axes.set_xlabel("LLM")
    axes.set_ylabel("Words per answer")
    axes.grid(axis="y", alpha=0.3)

    figure.tight_layout()

    # Save to results
    figure.savefig(f"results/{title}.png")

    plt.close(figure)

# Part 1: write the style statistics, characteristic words, and answer length plot to results/.
def describe_llm_differences(data_path: str = DATA_PATH) -> None:

    print(f"[describe_llm_differences] Measuring style of every answer in {data_path}")

    df = pd.read_csv(data_path)

    compute_style_statistics(df).round(3).to_csv(STYLE_STATISTICS_PATH)

    find_characteristic_words(df).to_csv(CHARACTERISTIC_WORDS_PATH, index=False)

    create_answer_length_plot(df, "RQ4 Answer Length")

    print(f"[describe_llm_differences] Wrote {STYLE_STATISTICS_PATH}, {CHARACTERISTIC_WORDS_PATH}, and the answer length plot")

# --- Part 2: remove one signal at a time ---

# Keep only the first EQUAL_LENGTH_TOKENS tokens, removing differences in answer length.
def cut_to_equal_length(tokens: list[str]) -> list[str]:
    return tokens[:EQUAL_LENGTH_TOKENS]

# Remove every punctuation token, leaving only words.
def remove_punctuation(tokens: list[str]) -> list[str]:
    return [token for token in tokens if re.fullmatch(r"\w+", token)]

# Replace every content word with a placeholder, keeping function words and punctuation. Tests whether style alone, without topic words, identifies the LLM.
def keep_function_words_only(tokens: list[str]) -> list[str]:
    return [token if token in FUNCTION_WORDS or not re.fullmatch(r"\w+", token) else CONTENT_PLACEHOLDER for token in tokens]

# Replace every function word with a placeholder, keeping content words and punctuation. The reverse of keep_function_words_only.
def keep_content_words_only(tokens: list[str]) -> list[str]:
    return [FUNCTION_PLACEHOLDER if token in FUNCTION_WORDS else token for token in tokens]

# Each ablation's name and the transform applied to every answer's tokens before training and testing.
ABLATIONS: dict[str, Callable[[list[str]], list[str]]] = {
    "equal length": cut_to_equal_length,
    "no punctuation": remove_punctuation,
    "function words only": keep_function_words_only,
    "content words only": keep_content_words_only,
}

# Train a model on outputs with one signal removed, then test it on the test set.
# Uses the same seed, split, and training as research question 1, so its accuracy compares directly with the research question 1 outputs only accuracy.
def evaluate_ablation(build_model: Callable, model_name: str, ablation_name: str, training_epochs: int) -> tuple[float, list[int], list[int], list[str]]:

    # Seed for replicability
    torch.manual_seed(1)

    train_loader, val_loader, test_loader, vocab_size, num_labels, label_names = get_data_loaders("outputs", answer_transform=ABLATIONS[ablation_name])

    model = train_model(build_model(vocab_size, num_labels), train_loader, val_loader, training_epochs, model_name=model_name, condition=f"outputs: {ablation_name}")

    accuracy, predictions, gold = evaluate(model=model, eval_loader=test_loader)

    print(f"[evaluate_ablation] {model_name} with {ablation_name}: test accuracy {accuracy}")

    return accuracy, predictions, gold, label_names

# One row of the ablation results: accuracy, drop from the unablated model, and how often each LLM was identified.
def ablation_row(model_name: str, ablation_name: str, accuracy: float, baseline_accuracy: float, predictions: list[int], gold: list[int], label_names: list[str]) -> dict:
    recalls = per_label_recall(predictions, gold, label_names)
    return {
        "model": model_name,
        "removed signal": ablation_name,
        "accuracy": accuracy,
        "drop from baseline": baseline_accuracy - accuracy,
        **{f"{label_name} recall": recall for label_name, recall in recalls.items()},
    }

# Bar chart of test accuracy for the unablated model and each ablation.
def create_ablation_plot(ablation_results: pd.DataFrame, title: str) -> None:

    figure, axes = plt.subplots(figsize=(8, 5))

    bars = axes.bar(ablation_results["removed signal"], ablation_results["accuracy"], color="tab:blue")

    # Write the accuracy above each bar
    for bar, accuracy in zip(bars, ablation_results["accuracy"]):
        axes.text(bar.get_x() + bar.get_width() / 2, accuracy + 0.01, f"{accuracy:.0%}", ha="center")

    # Chance accuracy for 5 LLMs
    axes.axhline(0.2, color="gray", linestyle="--", label="chance (20%)")

    axes.set_ylim(0, 1)
    axes.set_title(title)
    axes.set_xlabel("Signal removed")
    axes.set_ylabel("Test accuracy")
    axes.legend()

    plt.setp(axes.get_xticklabels(), rotation=15, ha="right")

    figure.tight_layout()

    # Save to results
    figure.savefig(f"results/{title}.png")

    plt.close(figure)

def main() -> None:
    describe_llm_differences()

if __name__ == "__main__":
    main()
