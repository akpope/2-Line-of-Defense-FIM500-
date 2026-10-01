# 2-Line-of-Defense-FIM500

## Project 1: Credit Card Fraud Detection

Credit Card Fraud Detection data obtained from Kaggle:  
https://www.kaggle.com/datasets/uditjain13/credit-card-fraud-detection-2026/data

## Project 2: EUR/USD Foreign Exchange Rate Analysis

EUR/USD foreign exchange rate data obtained from Yahoo Finance using the following Python code:

```python
data = yf.download(
    "EURUSD=X",
    start="2021-09-21",
    end="2026-09-21",
    interval="1d",
    auto_adjust=False
)
