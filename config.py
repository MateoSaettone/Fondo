# ============================================
#          Define Tickers for Analysis
# ============================================

# List of tickers categorized by industry and index
tickers = [
    # Target of the model
    "SPY", 
    
    # Technology Sector
    "AAPL", "MSFT", "NVDA", "GOOGL", "INTC",
    "CSCO", "TXN", "IBM", "ORCL", "QCOM", "AMZN",
    "TSLA", "META",
    
    # Financial Sector
    "GS", "BAC", "C", "WFC", "MS", "AXP", "BRK-B", 
    "V", "MA", "JPM",
    
    # Healthcare Sector
    "UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", 
    "AMGN", "MDT", "CI", "CVS",
    
    # Indices
    "^OEX", "NDAQ"
]

# ============================================
#               Define Industries
# ============================================

industries = {
    "tech": ["AAPL", "MSFT", "NVDA", "GOOGL", "INTC", "CSCO", "TXN", "IBM", "ORCL", "QCOM", "AMZN", "TSLA", "META"],
    "fin": ["GS", "BAC", "C", "WFC", "MS", "AXP", "BRK-B", "V", "MA", "JPM"],
    "health": ["UNH", "JNJ", "PFE", "LLY", "ABBV", "MRK", "AMGN", "MDT", "CI", "CVS"]
}


records = 1429