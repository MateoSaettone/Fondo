def calculate_features(data):
    data['MA_10'] = data['Price'].rolling(window=10).mean()
    data['Volatility'] = data['Price'].rolling(window=10).std()
    data['Returns'] = data['Price'].pct_change()
    data['Target'] = data['Price'].shift(-1)  # Predict next day's price
    return data.dropna()

# Example usage
if __name__ == "__main__":
    import api_integration
    df = api_integration.fetch_historical_data("SPY")
    df = calculate_features(df)
    print(df.head())
