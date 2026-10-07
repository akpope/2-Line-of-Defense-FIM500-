import numpy as np
import pandas as pd
import joblib

my_model = joblib.load("my_model.pkl")
scaling = joblib.load("scaling.pkl")
median_income = joblib.load("median_income.pkl")


def predict_pd(new_data):
    
    # Select the variables used by the model
    x_new = new_data[[
        "MonthlyIncome",
        "DebtRatio",
        "NumberOfTimes90DaysLate",
        "RevolvingUtilizationOfUnsecuredLines",
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfDependents",
        "NumberOfOpenCreditLinesAndLoans",
        "NumberRealEstateLoansOrLines",
        "age"
    ]].copy()
    
    # Handle missing MonthlyIncome
    x_new["MonthlyIncome"] = x_new["MonthlyIncome"].fillna(median_income)
    x_new["NumberOfDependents"]=x_new["NumberOfDependents"].fillna(median_depend)
    x_new["RevolvingUtilizationOfUnsecuredLines"]=x_new["RevolvingUtilizationOfUnsecuredLines"].clip(upper=upper_bound)
    
    # Use the scaler that was fitted on the original training data
    x_new_scaled = scaling.transform(x_new)
    
    # Predict probability of serious delinquency
    pd_predictions = my_model.predict_proba(x_new_scaled)[:, 1]
    
    # Create output table
    pd_table = pd.DataFrame({
        "Borrower": np.arange(1, len(pd_predictions) + 1),
        "PD (%)": pd_predictions*100
    })
    
    return pd_table

