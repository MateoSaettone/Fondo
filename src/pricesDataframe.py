import yfinance as yf
import pandas as pd

def fetch_price_series(ticker, start_date="2009-01-01", end_date="2026-01-01"):
    """
    Downloads data for one ticker and returns a DataFrame with two columns:
    Date and the Ticker's Price (renamed to match the ticker).
    """
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False)

    # Reset index so Date is a column
    data.reset_index(inplace=True)

    # Prefer 'Close' as the "Price". If not found, try 'Adj Close'
    if "Close" in data.columns:
        data.rename(columns={"Close": "Price"}, inplace=True)
    elif "Adj Close" in data.columns:
        data.rename(columns={"Adj Close": "Price"}, inplace=True)
    else:
        raise ValueError(f"No 'Close' or 'Adj Close' column found for {ticker}.")

    # Keep only Date and Price
    data = data[["Date", "Price"]]

    # Rename "Price" to the ticker symbol
    data.rename(columns={"Price": ticker}, inplace=True)

    return data

if __name__ == "__main__":

    # Tickers you want to download
    tickers = [
        # Tech / Other
        "SPY", "AAPL", "MSFT", "NVDA", "GOOGL", "INTC",
        "CSCO", "TXN", "IBM", "ORCL", "QCOM", "AMZN",
        "TSLA", "META", 

        # Financials
        "GS", "BAC", "C", "WFC", "MS", "AXP", "BRK-B", "V", "MA", "JPM",

        # Healthcare
        "UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", "AMGN",
        "MDT", "CI", "CVS",

        #Index
        "OEX", "NDAQ"
    ]

    merged_df = None

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")

        # Fetch a DataFrame: columns -> ["Date", ticker]
        df_ticker = fetch_price_series(ticker)

        # Merge into a single DataFrame
        if merged_df is None:
            # First ticker - just initialize
            merged_df = df_ticker
        else:
            # Merge on 'Date', using an outer join to keep all dates
            merged_df = pd.merge(merged_df, df_ticker, on="Date", how="outer")

    # Sort by Date, just to be clean
    merged_df.sort_values("Date", inplace=True)

    # Save to one CSV file with each ticker in its own column
    merged_df.to_csv("data/all_prices.csv", index=False)
    print("All tickers' prices saved to all_prices.csv")

    # Remove the second line from the newly created CSV file
    csv_path = "data/all_prices.csv"
    with open(csv_path, "r") as f:
        lines = f.readlines()
    if len(lines) > 1:
        del lines[1]  # Removes line #2 (index 1)
    with open(csv_path, "w") as f:
        f.writelines(lines)
    print(f"Removed the second line from {csv_path}\n")

    print(merged_df)
