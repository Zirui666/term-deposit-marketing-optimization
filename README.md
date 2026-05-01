🌍 **Predictive Optimization for Term-Deposit Marketing: Maximizing Campaign Profitability**

Term-deposit business relies heavily on marketing and relational strength with the cutsomer. Given the heterogeneity nature of customer portraits faced by retail banks, there is an urgent need to implement targeted marketing to maintain high profitability and stable business performances. Despite the need for targeted engagement, many campaigns still rely on generic outreach instead of data-driven profitability metrics.

This project uses predictive analytics and decision optimization to evaluate and maximize the expected net revenue of direct marketing campaigns. We integrate customer demographics, financial profiles, and contact history from a Portuguese banking institution (2008–2010) to predict conversion probabilities, expected deposit amounts, and withdrawal rates.

We explore four core questions:

* Does utilizing flexible machine learning models (like Explainable Boosting Machines or Neural Networks) improve predictive performance over traditional statistical models (like Tobit and Logistic regression)?
* Can we effectively predict customer conversion and deposit volume by decoupling the binary subscription decision from the underlying financial commitment?
* How can we select an optimal subset of clients for follow-up contact under strict marketing budget and return-on-marketing constraints?
* How do institutional constraints shape the optimal Annual Percentage Rate (APR) offered to distinct customer segments?

Our results show that integrating predictive models within a mixed integer linear programming (MIP) framework produces actionable, revenue-maximizing execution lists, revealing that optimal APR offers depend heavily on aggregate resource constraints rather than just individual customer responsiveness.

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

📂 **Repository Structure**

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

## Conclusion

Although the predictive models successfully estimate deposit behaviors, the optimal marketing decision is driven primarily by the interaction between customer-level personalized features and institutional constraints.
Statistically significant shifts in pricing strategy occur simply by tightening the return-on-marketing (ROM) constraints, with campaign feasibility sharply declining as requirements become more aggressive.

This highlights an important strategic lever in practical banking operations:

A model may accurately predict customer behavior, but effective bank marketing design requires the joint calibration of budget limits, return targets, and pricing strategy, since treating these levers independently risks conflating constraint: driven changes in optimal policy with changes in underlying demand.

## Authors

- Tiange Yu
- Michelle Schultze
- Zirui Yan
- Mingsheng Wang
