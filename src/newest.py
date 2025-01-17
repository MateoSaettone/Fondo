import yfinance as yf
import pandas as pd
import numpy as np
import os

# ### DATA

# Import historical data from Yahoo Finance API
# Prices per industry (Tickers)
tickers = [
    # Target of the model
    "SPY",

    # Technology
    "AAPL", "MSFT", "NVDA", "GOOGL", "INTC",
    "CSCO", "TXN", "IBM", "ORCL", "QCOM", "AMZN",
    "TSLA", "META",

    # Financials
    "GS", "BAC", "C", "WFC", "MS", "AXP", "BRK-B",
    "V", "MA", "JPM",

    # Healthcare
    "UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK",
    "AMGN", "MDT", "CI", "CVS",

    # Index(s)
    "^OEX", "NDAQ"
]

# Define industries
industries = {
    "Technology": ["AAPL", "MSFT", "NVDA", "GOOGL", "INTC", "CSCO", "TXN", "IBM", "ORCL", "QCOM", "AMZN", "TSLA", "META"],
    "Financials": ["GS", "BAC", "C", "WFC", "MS", "AXP", "BRK-B", "V", "MA", "JPM"],
    "Healthcare": ["UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", "AMGN", "MDT", "CI", "CVS"]
}

# Empty dictionary to store tickers
data_dict = {}

# Import historical data for each ticker
for ticker in tickers:
    stock_data = yf.Ticker(ticker).history(start="2009-01-01", end="2026-01-01", interval="1d")[['Close']]
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.index.name = 'Date'
    stock_data.rename(columns={'Close': ticker}, inplace=True)
    data_dict[ticker] = stock_data

# Merge DataFrames into a single combined DataFrame
SPY_combined = pd.DataFrame(data_dict["SPY"].copy())
for ticker in tickers:
    if ticker != "SPY":
        SPY_combined = SPY_combined.join(data_dict[ticker], how="inner")

# Adjust dataset
df = SPY_combined
df.rename(columns={'SPY': 'TARGET'}, inplace=True)
df = df.dropna()

# ### ROLLING BETA FOR EACH INDUSTRY

# Function to calculate rolling beta as percentage change
def calculate_rolling_beta_pct_change(industry_tickers, df):
    industry_data = {"Date": df.index}
    for ticker in industry_tickers:
        stock_corr = df[[ticker, "TARGET"]].dropna()
        n_records = len(stock_corr)
        betas = []

        # Calculate rolling beta
        for i in range(n_records - 100 + 1):
            X = stock_corr.iloc[i:i+100][ticker]
            Y = stock_corr.iloc[i:i+100]["TARGET"]
            cov_XY = X.cov(Y)
            var_X = X.var()
            beta = cov_XY / var_X
            betas.append(beta)

        # Add rolling beta values to industry data
        beta_series = pd.Series([np.nan] * 99 + betas, index=stock_corr.index, name=f"{ticker}_BETA")
        
        # Calculate percentage change in betas
        beta_pct_change = beta_series.pct_change() * 100
        industry_data[f"{ticker}_BETA"] = beta_pct_change

    return pd.DataFrame(industry_data).set_index("Date")

# Calculate rolling beta percentage changes for each industry
tech_industry_betas_pct = calculate_rolling_beta_pct_change(industries["Technology"], df)
fin_industry_betas_pct = calculate_rolling_beta_pct_change(industries["Financials"], df)
health_industry_betas_pct = calculate_rolling_beta_pct_change(industries["Healthcare"], df)

# ### EXPORT TO CSV

# Define output directory
output_directory = "./industry_csv_files/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Export each industry's rolling beta percentage change DataFrame to a CSV file
tech_industry_betas_pct.to_csv(f"{output_directory}Technology_Betas_Pct_Change.csv")
fin_industry_betas_pct.to_csv(f"{output_directory}Financials_Betas_Pct_Change.csv")
health_industry_betas_pct.to_csv(f"{output_directory}Healthcare_Betas_Pct_Change.csv")

print("Rolling beta percentage change data exported successfully!")
