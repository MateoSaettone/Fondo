import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.stattools import adfuller

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------
def fetch_market_caps_batch(tickers):
    market_caps = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        cap = stock.info.get("marketCap")
        market_caps[ticker] = cap if cap else None
    return market_caps

def calculate_weights(capitalization):
    total_market_cap = sum(cap for cap in capitalization.values() if cap is not None)
    if total_market_cap == 0:
        return {ticker: 0 for ticker in capitalization}
    return {ticker: cap / total_market_cap for ticker, cap in capitalization.items() if cap is not None}

def weighted_average(df, companies, weights):
    """Compute weighted average for given tickers in df based on market-cap weights."""
    return sum(df[ticker] * weights[ticker] for ticker in companies if ticker in df.columns)


def flatten_dataframe(df, date_label="Date"):
    """
    - Resets multi-level index (if any) so that it becomes a single-level datetime index named 'Date'.
    - Flattens multi-level columns into a single level.
    """
    # If we have a MultiIndex in the index, reset it
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
        # Try to rename the first date-like column to "Date"
        # or handle multiple index levels if needed
        if date_label in df.columns:
            df.set_index(date_label, inplace=True)
        else:
            # If there's no explicit 'Date' column, just set the first column as index
            df.set_index(df.columns[0], inplace=True)

    # Flatten multi-level columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        # Build new column names by joining sub-levels with '_'
        new_cols = []
        for col_tuple in df.columns:
            # col_tuple can be like ("Close", "AAPL"), etc.
            new_col = "_".join([str(c) for c in col_tuple if c != ""])  # remove empties
            new_cols.append(new_col)
        df.columns = new_cols

    return df


# ----------------------------------------------------------------------
# Feature Engineering Functions
# ----------------------------------------------------------------------
def add_industry_weighted_averages(data, industry_tickers):
    """
    Create new columns for each industry's weighted-average close price
    and concatenate them to 'data'.
    """
    industry_averages = {}
    for industry, tickers in industry_tickers.items():
        caps = fetch_market_caps_batch(tickers)
        weights = calculate_weights(caps)
        industry_averages[f"{industry}_AVG"] = weighted_average(data, tickers, weights)
    return pd.concat([data, pd.DataFrame(industry_averages)], axis=1)

def filter_features_by_correlation_and_stationarity(data, target, threshold=0.05):
    """
    1. Correlate every column with 'target'
    2. Keep columns that pass the ADF stationarity test (p-value < threshold)
    """
    # .corr() drops non-numeric columns automatically
    corr_matrix = data.corr()
    
    if target not in corr_matrix.columns:
        raise ValueError(f"Column '{target}' not found after correlation. "
                         f"Check if '{target}' still exists in your DataFrame columns.")

    # Get absolute correlation values wrt target, then sort descending
    correlation = corr_matrix[target].abs().sort_values(ascending=False)

    selected_features = []
    for col in correlation.index:
        col_data = data[col].dropna()
        # Must have enough non-NaN data and variance > 0 to run ADF
        if len(col_data) < 2 or col_data.std() == 0:
            continue
        adf_pvalue = adfuller(col_data)[1]
        if adf_pvalue < threshold:
            selected_features.append(col)

    return selected_features

# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---------------------
    # 1) Fetch SPY data
    # ---------------------
    spy_data = yf.download("SPY", start="2010-01-01", end="2023-01-01", auto_adjust=True)
    # Make tomorrow's Close the 'TARGET'
    spy_data['TARGET'] = spy_data['Close'].shift(-1)
    # Keep only columns we need
    spy_data = spy_data[['Close', 'TARGET']].dropna()

    # Flatten in case columns or index is multi-level
    spy_data = flatten_dataframe(spy_data)

    # ---------------------
    # 2) Fetch industry data
    # ---------------------
    industries = {
        "INFTECH": ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META'],
        "FINANCIALS": ['JPM', 'BAC', 'GS', 'C', 'WFC'],
        "HEALTHCARE": ['JNJ', 'PFE', 'LLY', 'ABBV', 'MRK']
    }
    industry_tickers = [ticker for tickers in industries.values() for ticker in tickers]
    
    industry_data = yf.download(
        industry_tickers,
        start="2010-01-01",
        end="2023-01-01",
        auto_adjust=True
    )['Close']

    # Flatten multi-level for industry_data
    industry_data = flatten_dataframe(industry_data)

    # 3) Add weighted averages for each industry
    industry_data = add_industry_weighted_averages(industry_data, industries)
    # Flatten again in case the new columns introduced a new multi-level
    industry_data = flatten_dataframe(industry_data)

    # ---------------------
    # 4) Merge the Data
    # ---------------------
    # Use how="left" to keep all rows from spy_data (and thus preserve 'TARGET')
    spy_data = pd.merge(spy_data, industry_data, left_index=True, right_index=True, how="left")

    # ---------------------
    # 5) Percentage changes
    # ---------------------
    spy_data = spy_data.pct_change().dropna()

    # Double-check that 'TARGET' still exists
    if 'TARGET' not in spy_data.columns:
        raise ValueError("Column 'TARGET' disappeared. Possibly no overlapping rows or dropna removed it.")

    # ---------------------
    # 6) Feature Selection
    # ---------------------
    selected_features = filter_features_by_correlation_and_stationarity(spy_data, "TARGET")
    X = spy_data[selected_features]
    y = spy_data["TARGET"]

    # ---------------------
    # 7) Train/Test Split
    # ---------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

    # Add constant
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # ---------------------
    # 8) Train the Model
    # ---------------------
    model = sm.OLS(y_train, X_train).fit()
    print(model.summary())

    # ---------------------
    # 9) Predictions
    # ---------------------
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print(f"R-squared: {r2}")
    print(f"Mean Squared Error: {mse}")

    # Create DataFrame of predictions
    predictions = pd.DataFrame({
        "Date": y_test.index,
        "Actual": y_test.values,
        "Predicted": y_pred.values,
        "Difference": y_pred.values - y_test.values
    })
    # Shift by one day if you want next-day alignment
    predictions = predictions.shift(1).dropna()

    # ---------------------
    # 10) Save Output
    # ---------------------
    predictions.to_csv("predictions.csv", index=False)
    print("Predictions saved to predictions.csv.")
