import yfinance as yf
import pandas as pd 
import numpy as np
import statsmodels.api as sm

# ============================================
#     Function to Calculate Rolling Beta
# ============================================

def calculate_rolling_beta(industry_tickers, df):
    """
    Calculate rolling beta values for a list of tickers within an industry.

    Parameters:
    - industry_tickers: List of ticker symbols in the industry.
    - df: DataFrame containing percentage changes of tickers and the TARGET.

    Returns:
    - DataFrame with rolling beta values for each ticker.
    """
    industry_data = {"Date": df.index}
    
    for ticker in industry_tickers:
        # Drop rows with missing values for the current ticker and TARGET
        stock_corr = df[[ticker, "TARGET"]].dropna()
        n_records = len(stock_corr)
        betas = []

        # Calculate beta over rolling windows of 1400 days
        for i in range(n_records - 1400 + 1):
            window = stock_corr.iloc[i:i+1400]
            X = window[ticker]
            Y = window["TARGET"]
            cov_XY = X.cov(Y)
            var_X = X.var()
            beta = cov_XY / var_X
            betas.append(beta)

        # Create a Series with NaNs for the initial periods without enough data
        beta_series = pd.Series([np.nan] * 1399 + betas, index=stock_corr.index, name=f"{ticker}_BETA")
        industry_data[f"{ticker}_BETA"] = beta_series

    return pd.DataFrame(industry_data).set_index("Date")


# ============================================
#        Function to Normalize Betas
# ============================================

def normalize_betas(beta_df):
    """
    Normalize the beta values by dividing each by the sum of betas in the industry.

    Parameters:
    - beta_df: DataFrame containing beta values for an industry.

    Returns:
    - Normalized beta DataFrame.
    """
    beta_df['SUM_BETA'] = beta_df.sum(axis=1)
    
    for column in beta_df.columns[:-1]:  # Exclude 'SUM_BETA'
        beta_df[column] = beta_df[column] / beta_df['SUM_BETA']
    
    beta_df = beta_df.drop(columns=['SUM_BETA'])
    return beta_df

# ============================================
#    Function to Update Betas with Stock Prices
# ============================================

def update_betas_with_prices(beta_df, price_data):
    """
    Multiply normalized betas by corresponding stock prices and sum them.

    Parameters:
    - beta_df: DataFrame containing normalized betas.
    - price_data: DataFrame containing stock prices.

    Returns:
    - DataFrame with weighted betas and their sum.
    """
    updated_beta_df = beta_df.copy()
    
    for column in beta_df.columns:
        ticker = column.replace("_BETA", "")
        if ticker in price_data.columns:
            updated_beta_df[column] = beta_df[column] * price_data[ticker]
    
    # Calculate the sum of weighted betas
    updated_beta_df["SUM"] = updated_beta_df.sum(axis=1)
    return updated_beta_df