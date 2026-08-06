# Credit Scoring — Credit Risk Assessment

End-to-end data science project: predicting the probability of loan default, using the German Credit Data dataset.

## Objective

Build a complete pipeline — data cleaning, feature engineering, modeling, evaluation — to predict whether a client will repay their loan (good payer) or default (bad payer), following sound methodological practices for a real-world scoring project.

## Dataset

**Source**: [German Credit Data (with Credit Risk)](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk), an enriched version of the original UCI/Statlog dataset that includes the true target variable.

- 954 clients, 10 columns (9 features + 1 target)
- Target variable `Credit Risk`: `0` = good payer, `1` = default (remapped from 1/2)
- Realistic class imbalance: **~70% good payers / 30% defaults**

| Variable | Type | Description |
|---|---|---|
| Age | Numerical | Client's age |
| Sex | Categorical | Gender |
| Job | Categorical (encoded) | Skill level |
| Housing | Categorical | Housing type (own / rent / free) |
| Saving accounts | Categorical | Savings account level |
| Checking account | Categorical | Checking account level |
| Credit amount | Numerical | Requested loan amount |
| Duration | Numerical | Loan duration (months) |
| Purpose | Categorical | Purpose of the loan |

## Project pipeline

```
data/raw/               → Raw dataset
      │
      ▼ preprocess.py    (cleaning: dropping unused columns, handling missing values)
data/processed/clean.csv
      │
      ▼ features.py      (feature engineering: ratio, target remapping, encoding)
data/processed/features.csv
      │
      ▼ train_logistic.py / train_rf.py   (split, scaling, training, evaluation)
reports/                 (ROC curves, metrics)
```

## Key methodological point: detecting and fixing a data leakage issue

One of the core challenges of this project was identifying a **data leakage** problem:

1. The Kaggle dataset initially used contained **no real target column**. A target had been artificially fabricated from a threshold on `Credit amount` and `Duration` — two variables that were then reused as features. Result: a 0.998 ROC-AUC score, far beyond what the literature reports for this dataset (~0.75-0.80), a clear sign the model was "reading" a formula rather than an actual financial signal.
2. A second leak of the same kind was found in the training script itself (`Target_Class` derived from `Installment_Per_Month`).
3. The dataset was replaced with a version containing the real historical target (`Credit Risk`), and both faulty blocks were removed.
4. A more rigorous approach to missing values was also implemented: `Saving accounts` and `Checking account` had ~55% of their `NA`s dropped by an initial `dropna()`, causing a loss of 46% of the dataset's rows and distorting the class ratio (70/30 → 55/45). These `NA`s actually reflect the absence of that account type for the client, not missing data — they were therefore replaced with an explicit `"none"` category instead of being dropped.

After the fix, the logistic regression model achieves a ROC-AUC of **0.756**, consistent with published benchmarks for this dataset.

## Exploratory analysis — key findings

- **`Checking account = none`** is the most discriminative signal in the dataset (correlation of -0.33 with risk). Counter-intuitively, having no checking account is associated with a lower default rate (~12% vs. ~30% on average) — likely a sign of a stable financial situation that doesn't require day-to-day cash management, rather than banking exclusion.
- **`Duration`** and **`Credit amount`** are positively correlated with risk: longer and larger loans are associated with more defaults.
- **`Housing = own`** is associated with lower risk than renting.
- **`Age`** and **`Sex`** show weak discriminative power on their own.

## Modeling

Two models were trained and compared on the same train/test split (80/20, stratified, `random_state=42`):

| Model | ROC-AUC | Notes |
|---|---|---|
| Logistic Regression | 0.756 | `StandardScaler` (fit on train only) + `class_weight='balanced'` |
| Random Forest (tuned) | **0.770** | `n_estimators=200`, `max_depth=6`, `min_samples_leaf=10`, `class_weight='balanced'` |

The unconstrained Random Forest (unlimited depth) achieved a lower AUC (0.74) than logistic regression — a classic case of overfitting on a small dataset (954 rows). Limiting tree depth and minimum leaf size allowed it to outperform logistic regression.

Both models converge on the same top features (`Checking account_none`, `Duration`, `Credit amount`), which reinforces confidence in these signals.

## Project structure

```
credit-scoring-project/
├── api/                  # (upcoming) FastAPI scoring API
├── data/
│   ├── raw/               # Raw dataset
│   └── processed/         # Cleaned data and features
├── notebooks/
│   └── 01_eda.ipynb       # Exploratory data analysis
├── src/
│   ├── preprocess.py      # Cleaning (ETL)
│   ├── features.py        # Feature engineering
│   ├── train_logistic.py  # Logistic regression training
│   └── train_rf.py        # Random Forest training
├── reports/                # Generated ROC curves
├── requirements.txt
└── README.md
```

## Setup and usage

```bash
pip install -r requirements.txt

python src/preprocess.py
python src/features.py
python src/train_logistic.py
python src/train_rf.py
```

## Next steps

- **SHAP**: fine-grained interpretability of Random Forest predictions, essential in credit scoring to justify individual decisions
- **FastAPI**: `/predict` endpoint for real-time scoring
- **Probability calibration** (Platt scaling / isotonic regression)
- **Docker** for containerization
- **Experiment tracking** (MLflow or a lightweight equivalent)

## Tech stack

Python · Pandas · Scikit-learn · Matplotlib · Git