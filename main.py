import yfinance as yf
import pandas as pd 
import numpy as np
import statsmodels.api as sm
from config import tickers, industries, records
from betas import calculate_rolling_beta, normalize_betas, update_betas_with_prices
from data import fetch_yahoo_data
from backtesting import perform_backtest

# ============================================
#             Virtual Environment Setup
# ============================================

# To activate the virtual environment, use the following command:
# venv\Scripts\activate

# ============================================
#        Import Historical Stock Data
# ============================================

# Initialize an empty dictionary to store data for each ticker

data_dict = fetch_yahoo_data(tickers)

# ============================================
#      Create Individual DataFrames for Each Ticker
# ============================================

ticker_dfs = {ticker: data_dict[ticker][[ticker]].copy() for ticker in tickers}

# ============================================
#              Merge All DataFrames
# ============================================

# Start with the SPY dataframe
SPY_combined = ticker_dfs['SPY'].copy()

# Join all other tickers to the SPY dataframe on the 'Date' index
for ticker in tickers:
    if ticker != "SPY":  
        SPY_combined = SPY_combined.join(ticker_dfs[ticker], how="inner")

# ============================================
#               Adjust the Dataset
# ============================================

# Rename 'SPY' column to 'TARGET' for clarity
df = SPY_combined.rename(columns={'SPY': 'TARGET'})

# Remove any rows with missing values
df = df.dropna()

# Create a copy of the dataframe for modeling
model_df = df.copy()

# ============================================
#             Import and Prepare SPY Data
# ============================================

# Initialize dictionary to store SPY data
spy_data_dict = {}

# Fetch historical data for SPY
stock_data = yf.Ticker("SPY").history(
    start="2013-01-02", 
    end="2026-01-01", 
    interval="1d", 
    auto_adjust=False
)
stock_data.index = stock_data.index.tz_localize(None)
stock_data.index.name = 'Date'
spy_data_dict["SPY"] = stock_data 

# Combine SPY data
SPY_combined = spy_data_dict["SPY"]

# Prepare the target dataset by removing nulls
target = SPY_combined.dropna()

# Calculate the target return as percentage change from Open to Close
target['TARGET'] = (target['Close'] / target['Open']) - 1

# Drop unnecessary columns to retain only the 'TARGET' column
target = target.drop(columns=[
    'Dividends', 'Volume', 'Stock Splits', 
    'Capital Gains', 'Low', 'High', 'Open', 'Close'
])

# Drop the first row to align with percentage changes
target = target.drop(df.index[0])

# ============================================
#        Convert Prices to Percentage Changes
# ============================================

# Calculate daily percentage changes for all tickers
df = df.pct_change().dropna() * 100  # Multiply by 100 to get percentage

# Add the 'TARGET' column to the dataframe
df[['TARGET']] = target[['TARGET']]

# ============================================
#        Calculate Rolling Betas for Industries
# ============================================

BETA_tech = calculate_rolling_beta(industries["tech"], df).dropna()
BETA_fin = calculate_rolling_beta(industries["fin"], df).dropna()
BETA_health = calculate_rolling_beta(industries["health"], df).dropna()

# ============================================
#           Normalize Betas for Each Industry
# ============================================

BETA_tech_normalized = normalize_betas(BETA_tech)
BETA_fin_normalized = normalize_betas(BETA_fin)
BETA_health_normalized = normalize_betas(BETA_health)

# ============================================
#      Apply Stock Prices to Normalized Betas
# ============================================

weighted_tech = update_betas_with_prices(BETA_tech_normalized, SPY_combined)
weighted_fin = update_betas_with_prices(BETA_fin_normalized, SPY_combined)
weighted_health = update_betas_with_prices(BETA_health_normalized, SPY_combined)

# ============================================
#           Create Final Regression Dataset
# ============================================

regression = pd.DataFrame({
    "TARGET": model_df["TARGET"],
    "TECH_INDUSTRY": weighted_tech["SUM"],
    "FIN_INDUSTRY": weighted_fin["SUM"],
    "HEALTH_INDUSTRY": weighted_health["SUM"],
    "OEX": model_df["^OEX"],
    "NDX": model_df["NDAQ"]
})

# ============================================
#              Shift the Target
# ============================================

# Shift the TARGET column by -1 to align predictions
regression['TARGET'] = regression['TARGET'].shift(-1)

# Drop any rows with missing values after shifting
regression = regression.dropna()

# ============================================
#             Adjust the DataFrame
# ============================================

# Further clean the regression DataFrame
regression = regression.dropna()
regression = regression.pct_change().dropna()  # Calculate percentage change again

# ============================================
#              Define X and Y for Regression
# ============================================

X = regression.drop(columns=['TARGET'])  # Predictor variables
Y = regression['TARGET']                # Response variable

# Add a constant term for the intercept in the regression model
X = sm.add_constant(X)

# ============================================
#           Initialize Variables for OLS
# ============================================

betas = []  # List to store regression coefficients

# ============================================
#    Train Initial OLS Model with First 1429 Records
# ============================================

initial_data = regression.iloc[:records]  # Select initial subset of data
X_initial = initial_data.drop(columns=['TARGET'])
Y_initial = initial_data['TARGET']
X_initial = sm.add_constant(X_initial)  # Add constant to initial predictors

# Fit the OLS regression model on the initial data
model_initial = sm.OLS(Y_initial, X_initial).fit()

# Store the initial coefficients
betas.append(model_initial.params)

# ============================================
#   Recalculate Betas for Each New Record
# ============================================

for i in range(records, len(regression)):  
    # Subset data up to the current index
    current_data = regression.iloc[:i+1]
    X_current = current_data.drop(columns=['TARGET'])
    Y_current = current_data['TARGET']
    X_current = sm.add_constant(X_current)  # Add constant term
    
    # Create logarithmic weights using log(x + 1)
    indices = np.arange(1, len(current_data) + 1) 
    weights = np.log1p(indices)  
    
    # Fit Weighted Least Squares (WLS) regression with weights
    model_current = sm.WLS(Y_current, X_current, weights=weights).fit()  
    
    # Store the updated coefficients
    betas.append(model_current.params)

# ============================================
#              Convert Betas to DataFrame
# ============================================

betas_df = pd.DataFrame(betas)

# ============================================
#        Select Last 200 Rows for Regression
# ============================================

regression = regression.iloc[-201:]

# ============================================
#        Merge Regression with Betas Data
# ============================================

# Reset index to convert 'Date' from index to a column
regression_reset = regression.reset_index()

# Add the 'const' column from betas_df to the regression DataFrame
regression_reset['const'] = betas_df['const']

# Identify common columns between regression_reset and betas_df
common_columns = regression_reset.columns.intersection(betas_df.columns)

# Multiply the common columns by their corresponding betas
for col in common_columns:
    regression_reset[col] = regression_reset[col] * betas_df[col].values

# Set 'Date' as the index of the regression DataFrame
regression_reset.set_index('Date', inplace=True)

# Remove any rows with missing values
regression_reset = regression_reset.dropna()

# Update the regression DataFrame with the processed data
regression = regression_reset

# ============================================
#        Finalize Prediction Column
# ============================================

# Remove the 'TARGET' column as it's no longer needed
regression.drop(columns=['TARGET'], inplace=True)

# Create a 'PRED' column as the sum of all weighted predictors
regression['PRED'] = regression.sum(axis=1)

# Keep only the 'PRED' column for predictions
regression = regression[['PRED']]

# Remove any remaining missing values
regression = regression.dropna()

# ============================================
#         Prepare Predictions for Next Day
# ============================================

# Create an empty row with index '2025-01-17' for future prediction
regression.loc[pd.to_datetime('2025-01-17')] = None

# Shift the 'PRED' column by 1 to align predictions with the same date
regression['PRED'] = regression['PRED'].shift(1)

# Drop rows with missing values after shifting
regression = regression.dropna()

# ============================================
#           Save Predictions to CSV
# ============================================

regression.to_csv("PREDICTIONS.csv")

# ============================================
#              Load and Process Predictions
# ============================================

# Load the predictions from the CSV file
df_predictions = pd.read_csv('PREDICTIONS.csv')

# Rename the unnamed index column to 'Date' and convert to datetime
df_predictions.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
df_predictions['Date'] = pd.to_datetime(df_predictions['Date'])

# Set 'Date' as the index of the predictions DataFrame
df_predictions.set_index('Date', inplace=True)

# ============================================
#             Re-import SPY Data
# ============================================

# Define tickers for SPY analysis
spy_tickers = ["SPY"]

# Initialize dictionary to store SPY data
spy_data_dict = {}

# Fetch historical data for SPY
for ticker in spy_tickers:
    stock_data = yf.Ticker(ticker).history(
        start="2009-01-01", 
        end="2026-01-01", 
        interval="1d", 
        auto_adjust=False
    )
    stock_data.index = stock_data.index.tz_localize(None)
    stock_data.index.name = 'Date'
    spy_data_dict[ticker] = stock_data

# Combine SPY data
SPY_combined = spy_data_dict["SPY"]

# Prepare the dataset and remove any missing values
df = SPY_combined.rename(columns={'SPY': 'TARGET'}).dropna()

# Create a copy of the dataframe for modeling
model_df = df.copy()

# ============================================
#            Merge Final DataFrame
# ============================================

# Join the model dataframe with the predictions on the 'Date' index
df_final = model_df.join(df_predictions, how='inner')  

# ============================================
#      Calculate Prediction for the Same Date
# ============================================

# Calculate the predicted price for the same date
df_final["PRED_SAME_DATE"] = (df_final["PRED"] + 1) * df_final["Open"]

# Drop unnecessary columns to retain only relevant data
df_final = df_final.drop(columns=[
    'Dividends', 'Volume', 'Stock Splits', 
    'Capital Gains', 'PRED'
])

# Remove any remaining missing values
df_final = df_final.dropna()

# ============================================
#             Display Final DataFrame
# ============================================

# Display the final dataframe (optional, can be removed in production)
# print(df_final)

# ============================================
#           Add Trading Order Column
# ============================================

# Determine trading orders based on predictions
df_final['Order'] = df_final.apply(
    lambda row: 'CALL' if row['PRED_SAME_DATE'] > row['Open'] else 'PUT', axis=1
)

# ============================================
#        Perform Backtesting
# ============================================

# Perform backtesting and get average efficiency and PnL
average_efficiency, average_pnl = perform_backtest(df_final, filepath='BT_RESULTS.csv')

# Print the results
print(f"Average Efficiency: {average_efficiency:.2f}")
print(f"Average PnL: {average_pnl:.2f}")