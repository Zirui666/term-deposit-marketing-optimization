# Predictive Optimization for Term-Deposit Marketing

This repository contains the analytical notebook for an IDS 586 project on optimizing term-deposit marketing decisions for a banking campaign. The workflow combines predictive modeling with a mixed-integer optimization framework to support customer-level APR and phone-call decisions.

## Project Overview

The project follows a two-stage framework:

1. **Stage I: Predictive modeling**
   - Preprocesses customer-level marketing campaign data.
   - Estimates synthetic deposit amount using a Tobit-style censored regression framework.
   - Estimates conversion probability using logistic regression and machine learning alternatives.
   - Segments customers into liquidity-risk groups and estimates expected withdrawal risk.

2. **Stage II: Decision optimization**
   - Builds a customer-offer decision grid.
   - Computes expected EBIT for each feasible APR and call/no-call decision.
   - Solves a mixed-integer programming problem to select the optimal offer for each customer.
   - Produces sensitivity analysis across budget and return-on-marketing constraints.

## Repository Structure

```text
term-deposit-marketing-optimization/
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
│   └── IDS_586_term_deposit_marketing.ipynb
└── src/
    └── README.md
```

## Data

Put the processed dataset in the `data/` folder with this exact filename:

```text
bank_marketing_refined_with_synth_apr (cleaned records).csv
```

The notebook is configured to read the dataset from `data/` and save generated tables, figures, and model outputs to `outputs/`.

## How to Run

1. Clone or download this repository.
2. Place the processed CSV file in the `data/` folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Open and run:

```text
notebooks/IDS_586_term_deposit_marketing.ipynb
```

## Main Outputs

Typical outputs include:

- `missing_by_column.csv`
- `liquidity_risk_profiles.csv`
- `synthesized_deposit_summary.csv`
- `deposit_model_comparison.csv`
- `conversion_model_evaluation_full.csv`
- `decision_grid_fixed.csv`
- `FINAL_OPTIMAL_OFFERS.csv`
- APR assignment and sensitivity-analysis plots

## Notes

The cost parameters used in the optimization are calibrated campaign-level marginal costs. They should be interpreted as modeling assumptions for decision optimization rather than universal accounting benchmarks.

## Authors

- Tiange Yu
- Michelle Schultze
- Zirui Yan
- Mingsheng Wang
