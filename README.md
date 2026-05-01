# Predictive Optimization for Term-Deposit Marketing

This repository is organized to match the project code structure: Stage I generates the prediction outputs and decision grid, and Stage II solves the optimization problem.

## Required Data Files

Place **both** CSV files in the `data/` folder before running the code:

```text
data/
├── bank-full.csv
└── bank_marketing_refined_with_synth_apr (cleaned records).csv
```

The original Stage I code reads both files. The main modeling pipeline then uses the processed dataset:

```python
df_old = pd.read_csv(old_path, sep=";")
df_new = pd.read_csv(new_path)
df = df_new.copy()
```

Therefore, `bank-full.csv` is required for the script to run without a file-not-found error, even though the downstream modeling steps are based on the processed dataset.

## Repository Structure

```text
term-deposit-marketing-optimization-strict-code/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── outputs/
│   └── README.md
├── figures/
│   └── README.md
├── notebooks/
│   └── run_project_pipeline.ipynb
└── src/
    ├── __init__.py
    ├── stage_1.py
    └── stage_2.py
```

## How to Run

### Option 1: Run the notebook

Open:

```text
notebooks/run_project_pipeline.ipynb
```

Then run all cells.

### Option 2: Run from terminal

From the repository root, run:

```bash
python -m src.stage_1
```

## Outputs

Generated outputs are saved to the `outputs/` folder, including summary tables, model metrics, decision grids, optimal offers, and figures.

## Installation

```bash
pip install -r requirements.txt
```

If the `interpret` package fails to install, use Python 3.10--3.12.

## Authors

- Tiange Yu
- Michelle Schultze
- Zirui Yan
- Mingsheng Wang
