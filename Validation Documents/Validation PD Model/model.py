import numpy as np
import pandas as pd
import joblib

# VALIDATOR NOTE (F1, High): the original file referenced `median_depend` and `upper_bound`,
# which were never defined or saved, so predict_pd() raised NameError. Both are now loaded from
# pickles that model.ipynb (cell 1) now saves alongside the others.
my_model = joblib.load("my_model.pkl")
scaling = joblib.load("scaling.pkl")
median_income = joblib.load("median_income.pkl")
median_depend = joblib.load("median_depend.pkl")   # added by validator
upper_bound = joblib.load("upper_bound.pkl")       # added by validator

FEATURES = [
    "MonthlyIncome",
    "DebtRatio",
    "NumberOfTimes90DaysLate",
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "age",
]  # ORDER MATTERS: must equal the column order used when the scaler/model were fitted


def predict_pd(new_data):

    # Fail clearly if a required column is missing (original raised a bare KeyError)
    missing = [c for c in FEATURES if c not in new_data.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Select the variables used by the model
    x_new = new_data[FEATURES].copy()

    # Handle missing values with TRAINING-set medians (good: no leakage from scoring data)
    x_new["MonthlyIncome"] = x_new["MonthlyIncome"].fillna(median_income)
    x_new["NumberOfDependents"] = x_new["NumberOfDependents"].fillna(median_depend)

    # VALIDATOR NOTE (F2, High): this cap is applied at scoring time, but in model.ipynb the
    # equivalent clip was applied to `training_data` AFTER x_train/x_test were built, so the model
    # was fitted on UNCLIPPED utilization. Training and scoring are inconsistent. The developer must
    # either (a) clip x_train/x_test before fitting (preferred) or (b) remove this line.
    # Also: MonthlyIncome was clipped at the 75th percentile in the notebook but is NOT clipped here.
    x_new["RevolvingUtilizationOfUnsecuredLines"] = x_new["RevolvingUtilizationOfUnsecuredLines"].clip(upper=upper_bound)

    # VALIDATOR NOTE (F6, Medium): delinquency counts contain sentinel codes 96/98 (214 rows in
    # cs-test) that are treated as real counts. Not handled by the developer.

    # Use the scaler that was fitted on the original training data
    x_new_scaled = scaling.transform(x_new)

    # Predict probability of serious delinquency
    pd_predictions = my_model.predict_proba(x_new_scaled)[:, 1]

    # Create output table
    # VALIDATOR NOTE: 'Borrower' is a row counter, not an ID; use new_data.index if IDs matter.
    pd_table = pd.DataFrame({
        "Borrower": np.arange(1, len(pd_predictions) + 1),
        "PD (%)": pd_predictions * 100
    })

    return pd_table
