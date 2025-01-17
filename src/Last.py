# ### LIBRARIES
import os
import yfinance as yf
import pandas as pd
import numpy as np

# ### DATA

# Define stock tickers
tickers = [
    "SPY",  # Target of the model
    
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

# Empty dictionary to store ticker data
data_dict = {}

# Import historical data for each ticker
for ticker in tickers:
    stock_data = yf.Ticker(ticker).history(start="2009-01-01", end="2026-01-01", interval="1d")[['Close']]
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.index.name = 'Date'
    stock_data.rename(columns={'Close': ticker}, inplace=True)
    data_dict[ticker] = stock_data

# Merge dataframes
SPY_combined = data_dict["SPY"].copy()
for ticker in tickers:
    if ticker != "SPY":
        SPY_combined = SPY_combined.join(data_dict[ticker], how="inner")

# Adjust dataset
df = SPY_combined
df.rename(columns={'SPY': 'TARGET'}, inplace=True)
df = df.dropna()

# Convert prices to percentage changes
df = df.pct_change()
df = df.dropna()
df = df * 100

# Define industries
industries = {
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "INTC", "CSCO", "TXN", "IBM", "ORCL", "QCOM", "AMZN", "TSLA", "META"],
    "fin": ["GS", "BAC", "C", "WFC", "MS", "AXP", "BRK-B", "V", "MA", "JPM"],
    "health": ["UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", "AMGN", "MDT", "CI", "CVS"]
}

# Function to calculate rolling beta for multiple stocks
def calculate_rolling_beta(industry_tickers, df):
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

        # Create DataFrame with rolling betas
        beta_series = pd.Series([np.nan] * 99 + betas, index=stock_corr.index, name=f"{ticker}_BETA")
        industry_data[f"{ticker}_BETA"] = beta_series

    return pd.DataFrame(industry_data).set_index("Date")

# Calculate rolling betas for each industry
BETA_tech = calculate_rolling_beta(industries["tech"], df).dropna()
BETA_fin = calculate_rolling_beta(industries["fin"], df).dropna()
BETA_health = calculate_rolling_beta(industries["health"], df).dropna()

# Normalize betas by dividing by the sum of all betas for the respective industry
def normalize_betas(beta_df):
    beta_df['SUM_BETA'] = beta_df.sum(axis=1)
    for column in beta_df.columns[:-1]:  # Exclude SUM_BETA column
        beta_df[column] = beta_df[column] / beta_df['SUM_BETA']
    beta_df = beta_df.drop(columns=['SUM_BETA'])
    return beta_df

# Normalize each industry's betas
BETA_tech_normalized = normalize_betas(BETA_tech)
BETA_fin_normalized = normalize_betas(BETA_fin)
BETA_health_normalized = normalize_betas(BETA_health)

# Multiply normalized betas by stock prices
def update_betas_with_prices(beta_df, price_data):
    updated_beta_df = beta_df.copy()
    for column in beta_df.columns:
        ticker = column.replace("_BETA", "")
        if ticker in price_data.columns:
            updated_beta_df[column] = beta_df[column] * price_data[ticker]
    updated_beta_df["SUM"] = updated_beta_df.sum(axis=1)
    return updated_beta_df

# Apply stock prices to normalized betas
BETA_tech_updated = update_betas_with_prices(BETA_tech_normalized, SPY_combined)
BETA_fin_updated = update_betas_with_prices(BETA_fin_normalized, SPY_combined)
BETA_health_updated = update_betas_with_prices(BETA_health_normalized, SPY_combined)

# Define output directory
output_directory = "./updated_industry_csv_files/"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Export updated betas to new CSV files
BETA_tech_updated.to_csv(f"{output_directory}Technology_Updated.csv")
BETA_fin_updated.to_csv(f"{output_directory}Financials_Updated.csv")
BETA_health_updated.to_csv(f"{output_directory}Healthcare_Updated.csv")

print("Updated betas with stock prices exported to new CSV files successfully.")
