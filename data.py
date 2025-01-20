import yfinance as yf
import pandas as pd 
import numpy as np
import statsmodels.api as sm



def fetch_yahoo_data(tickers, start="2009-01-01", end="2026-01-01", interval="1d", auto_adjust=False):
    data_dict = {}

    for ticker in tickers:
        stock_data = yf.Ticker(ticker).history(
            start="2009-01-01", 
            end="2026-01-01", 
            interval="1d", 
            auto_adjust=False
        )[['Close']]  # Select only the 'Close' price
        
        # Remove timezone information from the index
        stock_data.index = stock_data.index.tz_localize(None)
        stock_data.index.name = 'Date'
        
        # Rename the 'Close' column to the ticker symbol
        stock_data.rename(columns={'Close': ticker}, inplace=True)
        
        # Store the processed data in the dictionary
        data_dict[ticker] = stock_data

    return data_dict