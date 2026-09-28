import numpy as np
import pandas as pd
import joblib

my_model = joblib.load("my_model.pkl")
scaling = joblib.load("scaling.pkl")
median_income = joblib.load("median_income.pkl")


def predict_pd(new_data):

    features = [
        "MonthlyIncome",
        "DebtRatio",
        "NumberOfTimes90DaysLate",
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse"
    ]

    x_new = new_data[features].copy()

    x_new["MonthlyIncome"] = x_new["MonthlyIncome"].fillna(median_income)

    x_new_scaled = scaling.transform(x_new)

    pd_predictions = my_model.predict_proba(x_new_scaled)[:, 1]

    pd_table = pd.DataFrame({
        "Borrower": np.arange(1, len(pd_predictions) + 1),
        "PD (%)": pd_predictions*100
    })

    return pd_table
