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

### GPU (optional)

The CNN and RNN train on an NVIDIA GPU automatically when PyTorch can see one,
and fall back to the CPU otherwise. Every training run prints
`[train_model] Training on device: ...` when it starts.

On Windows, `pip install -r requirements.txt` installs the CPU-only build of
PyTorch. To use a GPU, reinstall PyTorch with CUDA support using the command
from the [PyTorch install selector](https://pytorch.org/get-started/locally/),
then check that it worked:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Data

This project uses the TXD-22 dataset (Iqbal & Abul Kashem, 2026), available on
[Kaggle](https://www.kaggle.com/datasets/mdsadiqiqbal/txd-22-a-large-scale-benchmark-dataset)

The raw file is too large for GitHub, so it is not included in the repository.

1. Download the CSV from Kaggle.
2. Save it as `data/TXD-22 A Large-Scale Benchmark Dataset.csv`
3. Run `python data_pipeline.py` to prepare data for our machine learning pipeline.
4. The filtered and labeled data will be output to `data\TXD-22 Filtered and Labeled.csv`

### Data fixes

We keep the ChatGPT, Claude, Mistral AI, Copilot, and Qwen sources. Each source has
3,000 rows: 30 subjects with 100 questions each, in the same order, except for the issues below. The dataset
has no subject column, so `data_pipeline.py` assigns each row's subject
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

## Running the experiments

Run every experiment from the repository root after preparing the data:

```bash
python research_questions.py
```

This trains the CNN and RNN for research questions 1, 2, and 3, runs the
research question 4 analysis, and writes the results to `results/`. Training uses the same train/validation/test split for
both models (`data_splitter.py`), split by question so all five models' answers
to a question are always in the same split.

| File | Contents |
| --- | --- |
| `accuracies.csv` | Test accuracy for every run: RQ1/RQ2 (outputs only, inputs only, both) and RQ3 (every training domain → testing domain pair) |
| `rq3_recall.csv` | RQ3: how often each LLM was identified correctly for every training domain → testing domain pair |
| `training_history.csv` | Training loss, validation accuracy, and time for every epoch of every run |
| `training_summary.csv` | Parameter count, best epoch, and total training time for every run |
| `CNN/RNN ... Confusion Matrix.png` | RQ1/RQ2 confusion matrices for each model and condition |
| `CNN/RNN Cross Domain Accuracy.png` | RQ3 grid: rows are the training domain, columns the testing domain |
| `CNN/RNN Training Curves.png` | Training loss and validation accuracy per epoch for the RQ1/RQ2 runs |
| `rq4_style_statistics.csv` | RQ4: average length, punctuation, formatting, and vocabulary measures for each LLM |
| `rq4_characteristic_words.csv` | RQ4: words each LLM uses much more often than the others |
| `rq4_ablation.csv`, `RQ4 CNN Ablation Accuracy.png` | RQ4: CNN accuracy and per-LLM recall after removing one signal (length, punctuation, content words, or function words) |
| `RQ4 Answer Length.png` | RQ4: distribution of answer length for each LLM |

`python rq4.py` writes only the RQ4 statistics, characteristic words, and answer
length plot, without training anything.

### Code layout

| File | Purpose |
| --- | --- |
| `data_pipeline.py` | Cleans the raw dataset and adds subject and domain labels |
| `data_splitter.py` | Train/validation/test split shared by both models |
| `text_pipeline.py` | Tokenizing, vocabulary, and PyTorch data loaders |
| `training.py` | Training loop, evaluation, and training logs shared by both models |
| `cnn.py`, `rnn.py` | The CNN and RNN (bidirectional LSTM) models |
| `analysis.py` | Confusion matrices, RQ3 grid, recall, and training curve plots |
| `rq4.py` | RQ4 style statistics, characteristic words, and signal removal experiments |
| `research_questions.py` | Runs every experiment and writes `results/` |
