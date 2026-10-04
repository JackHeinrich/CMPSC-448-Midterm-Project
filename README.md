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

This project uses the TXD-22 dataset (Iqbal & Abul Kashem, 2026), avialable on
[Kaggle](https://www.kaggle.com/datasets/mdsadiqiqbal/txd-22-a-large-scale-benchmark-dataset)
The raw file is too large for GitHub, so it is not included in the repository.

1. Download the CSV from Kaggle.
2. Save it as `data/TXD-22 A Large-Scale Benchmark Dataset.csv`
3. Run `data/filter_and_label_data` to prepare data for our machine learning pipeline.
