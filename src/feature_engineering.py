import pandas as pd

def calculate_moving_average(data, column, window):
    """
    Calculate a moving average for a given column.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column to calculate the moving average on.
        window (int): The size of the moving window.

    Returns:
        pd.Series: The moving average values.
    """
    return data[column].rolling(window=window).mean()

def calculate_volatility(data, column, window):
    """
    Calculate the rolling volatility (standard deviation) for a given column.

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column to calculate the volatility on.
        window (int): The size of the rolling window.

    Returns:
        pd.Series: The rolling volatility values.
    """
    return data[column].rolling(window=window).std()

def calculate_returns(data, column):
    """
    Calculate the percentage change in a given column (returns).

    Parameters:
        data (pd.DataFrame): The input DataFrame.
        column (str): The column to calculate the percentage change on.

    Returns:
        pd.Series: The percentage change values.
    """
    return data[column].pct_change()

def add_features(data):
    """
    Add features (moving averages, volatility, returns) to the dataset.

    Parameters:
        data (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with additional feature columns.
    """
    data['MA_10'] = calculate_moving_average(data, 'Price', 10)
    data['MA_50'] = calculate_moving_average(data, 'Price', 50)
    data['Volatility'] = calculate_volatility(data, 'Price', 10)
    data['Returns'] = calculate_returns(data, 'Price')
    return data.dropna()

if __name__ == "__main__":
    # Import fetch_historical_data from api_integration
    from pricesDataframe import fetch_historical_data

    # Fetch historical data
    df = fetch_historical_data("SPY", "2009-01-01", "2026-01-01")
    
    # Add features
    df = add_features(df)

    # Save results for debugging
    df.to_csv("data/sp500_features.csv", index=False)
    print(df.head())
    print("Features saved to data/sp500_features.csv")
