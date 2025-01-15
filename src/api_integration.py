import yfinance as yf
import pandas as pd

"""
Fetch historical data for a given ticker symbol from Yahoo Finance, ensuring
that the final columns are in the order Date, Price, High, Low, Open, and Volume.
'Price' will be taken from the 'Adj Close' column.

Parameters:
    ticker (str): Stock ticker symbol (e.g., "SPY").
    start_date (str): Start date in YYYY-MM-DD format.
    end_date (str): End date in YYYY-MM-DD format.

Returns:
    pandas.DataFrame: DataFrame containing historical stock data with columns:
                        Date, Price, High, Low, Open, Volume.
"""

def fetch_historical_data(ticker, start_date="2010-01-01", end_date="2023-01-01"):
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=True)
    data.reset_index(inplace=True)
    data.rename(columns={"Close": "Price"}, inplace=True)
    data = data[["Date", "Price", "High", "Low", "Open", "Volume"]]
    return data

if __name__ == "__main__":
    df = fetch_historical_data("SPY", "2010-01-01", "2023-01-01")
    print(df.head())
    df.to_csv("data/sp500_historical.csv", index=False)
    print("Data saved to data/sp500_historical.csv")
