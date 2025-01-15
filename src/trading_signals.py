import pandas as pd
import numpy as np

# Load predictions (adjust this path to match your predictions file)
predictions = pd.read_csv("data/predictions.csv")

# Ensure the data types are correct
predictions['Actual'] = predictions['Actual'].astype(float)
predictions['Predicted'] = predictions['Predicted'].astype(float)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

def calculate_difference(predictions):
    """
    Calculate the relative difference between predicted and actual values.
    """
    predictions['Difference'] = (predictions['Predicted'] - predictions['Actual']) / predictions['Actual']

def generate_signals(predictions, threshold=0.01):
    """
    Generate trading signals (BUY, SELL, HOLD) based on predictions.
    """
    predictions['Signal'] = 'HOLD'  # Default to HOLD

    # BUY Signal: Predicted > Actual by more than threshold
    predictions.loc[predictions['Difference'] > threshold, 'Signal'] = 'BUY'

    # SELL Signal: Predicted < Actual by more than threshold (negative difference)
    predictions.loc[predictions['Difference'] < -threshold, 'Signal'] = 'SELL'

def add_moving_average_signals(predictions, ma_window=5, threshold=0.01):
    """
    Add moving average-based signals for additional trend-based analysis.
    """
    predictions['Moving_Avg'] = predictions['Actual'].rolling(window=ma_window).mean()
    predictions['Trend_Signal'] = 'HOLD'  # Default to HOLD

    # Trend-based BUY Signal
    predictions.loc[
        (predictions['Predicted'] > predictions['Moving_Avg'] * (1 + threshold)),
        'Trend_Signal'
    ] = 'BUY'

    # Trend-based SELL Signal
    predictions.loc[
        (predictions['Predicted'] < predictions['Moving_Avg'] * (1 - threshold)),
        'Trend_Signal'
    ] = 'SELL'

# ----------------------------------------------------------------------
# Main Script
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Calculate the difference
    calculate_difference(predictions)

    # Generate trading signals
    threshold = 0.01  # Set threshold for BUY/SELL signals (1%)
    generate_signals(predictions, threshold=threshold)

    # Add optional moving average-based signals
    ma_window = 5  # Set moving average window (e.g., 5 days)
    add_moving_average_signals(predictions, ma_window=ma_window, threshold=threshold)

    # Save the results to a new CSV file
    output_path = "data/predictions_with_signals.csv"
    predictions.to_csv(output_path, index=False)


    # Print sample results
    print(predictions.head(20))
