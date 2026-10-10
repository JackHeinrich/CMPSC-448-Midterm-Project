# CMPSC-448-Midterm-Project

Midterm project for CMPSC 448.

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Data

This project uses the TXD-22 dataset (Iqbal & Abul Kashem, 2026), available on
[Kaggle](https://www.kaggle.com/datasets/mdsadiqiqbal/txd-22-a-large-scale-benchmark-dataset)

The raw file is too large for GitHub, so it is not included in the repository.

1. Download the CSV from Kaggle.
2. Save it as `data/TXD-22 A Large-Scale Benchmark Dataset.csv`
3. Run `filter_and_label_data.py` to prepare data for our machine learning pipeline.
4. The filtered and labeled data will be output to `data\TXD-22 Filtered and Labeled.csv`

### Data fixes

We keep the ChatGPT, Claude, Mistral AI, Copilot, and Qwen sources. Each source has
3,000 rows: 30 subjects with 100 questions each, in the same order, except for the issues below. The dataset
has no subject column, so `filter_and_label_data.py` assigns each row's subject
from its position.

- **Claude row order.** In the raw file, Claude's last Gender Studies question
  ("How can men and women collaborate for gender justice?") is at position 2899,
  the end of the Agriculture block, instead of position 2799. This shifts
  Claude's Agriculture questions back by one row. If left alone, 101 Claude rows
  would be paired with the wrong question wording, and two rows would get the
  wrong subject. The script moves that row back to position 2799. After the fix,
  every question lines up across all five sources.
- **Questions with mismatched answers dropped.** In some rows, the answer was
  written for a different question, usually a neighbouring one in the same
  subject. This happens in runs where one answer is duplicated and the rows after
  it are shifted by one. It affects every kept source, including ChatGPT (about
  50 rows) and Qwen (about 70 rows). We found these rows by comparing each
  answer's vocabulary with the other models' answers to every question in the
  same subject, and flagging answers that match another question more closely
  than their own.
  - Some flags are false alarms, where two questions are very similar, and some
    real errors may be missed. Checking every row by hand was not practical.
  - The 138 flagged question ids (`MISMATCHED_QUESTION_IDS`) are dropped from
    all five sources, so every source keeps the same 2,862 questions. That
    removes 690 of 15,000 rows (4.6%). The smallest subject keeps 76 of 100
    questions.
  - Dropping a question from every source, not just the source with the error,
    keeps the classes balanced. It also means input-only classification (RQ2)
    can't learn from which questions are missing for a given source.
  - The errors don't affect the `Source` labels: a shifted answer was still
    written by the model it is labeled as. They only affect which question an
    answer is paired with, which matters for the input + output experiments in
    RQ2.
- **Questions with cut-off answers dropped.** Five answers end mid-sentence
  (ChatGPT 713 and 1759, Copilot 715, Qwen 650 and 1052). Those 5 question ids
  (`CUT_OFF_QUESTION_IDS`) are dropped from all five sources, removing 25 more
  rows and leaving 2,857 questions per source. This step is its own function
  (`drop_cut_off_questions`), so it can be turned off on its own.

### Note for developers

Repository structure was flattened to make it easier for us to run python files that import functions from other files.
Neural network implementions will go in `/cnn.py` and `/rnn.py` respectively.

### Data collection

`/research_questions.py` is the entry point that will invoke all of our implementations to train the neural networks and export the results.
