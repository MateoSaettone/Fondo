import yfinance as yf
import pandas as pd

def debug_ticker_market_cap(ticker, start_date="2009-01-01", end_date="2026-01-01"):
    print(f"=== DEBUGGING TICKER: {ticker} ===")

    # 1. Download historical data
    data = yf.download(ticker, start=start_date, end=end_date, interval="1d", auto_adjust=False)
    print("\n1) Raw price data (first 5 rows):")
    print(data.head())

    # 2. Ticker info
    ticker_obj = yf.Ticker(ticker)
    info = ticker_obj.info
    print("\n2) Ticker info dictionary:")
    print(info)

    # 3. Shares outstanding
    shares_outstanding = info.get("sharesOutstanding", None)
    print(f"\n3) {ticker} sharesOutstanding: {shares_outstanding}")

    # 4. Check if data is empty
    if data.empty:
        print(f"\nNo data returned for {ticker} in the date range. Exiting debug.")
        return

    # 5. Handle multi-level columns (if yfinance returned multiple columns per ticker)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)
        print("\nDropped top-level from MultiIndex columns.")

    # 6. Calculate MarketCap if possible
    close_col = None
    if "Close" in data.columns:
        close_col = "Close"
    elif "Adj Close" in data.columns:
        close_col = "Adj Close"

    if close_col is None:
        print(f"\nNo 'Close' or 'Adj Close' column found for {ticker}. Exiting debug.")
        return

    # If sharesOutstanding is unavailable/None/0, we can set a default for debugging
    if not shares_outstanding:
        print(f"\n{ticker} has no valid sharesOutstanding. Using '1' just to show price data.")
        shares_outstanding = 1

    data["MarketCap"] = data[close_col] * shares_outstanding

    # 7. Keep only Date and MarketCap, rename MarketCap column to the ticker
    data.reset_index(inplace=True)
    data = data[["Date", "MarketCap"]]
    data.rename(columns={"MarketCap": ticker}, inplace=True)

    print(f"\n7) Market cap data (first 5 rows) for {ticker}:")
    print(data.head())

if __name__ == "__main__":
    # Debug one ticker at a time. Swap "AAPL" for any ticker you want to investigate.
    debug_ticker_market_cap("AAPL")
