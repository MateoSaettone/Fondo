# backtesting.py

import pandas as pd
import numpy as np

def determine_trading_orders(df_final):
    """
    Determine trading orders ('CALL' or 'PUT') based on predictions.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing 'PRED_SAME_DATE' and 'Open' columns.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'Order' column.
    """
    if 'PRED_SAME_DATE' not in df_final.columns or 'Open' not in df_final.columns:
        raise KeyError("DataFrame must contain 'PRED_SAME_DATE' and 'Open' columns to determine orders.")
    
    # Vectorized operation for better performance
    df_final['Order'] = np.where(df_final['PRED_SAME_DATE'] > df_final['Open'], 'CALL', 'PUT')
    
    return df_final

def calculate_precision(df_final):
    """
    Determine if the prediction was correct and add a 'Precision' column.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing 'Order', 'High', 'Low', and 'PRED_SAME_DATE' columns.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'Precision' column.
    """
    required_columns = ['Order', 'High', 'Low', 'PRED_SAME_DATE']
    for col in required_columns:
        if col not in df_final.columns:
            raise KeyError(f"Column '{col}' is missing from df_final.")
    
    # Vectorized calculation for better performance
    conditions = (
        ((df_final['Order'] == 'CALL') & (df_final['High'] > df_final['PRED_SAME_DATE'])) |
        ((df_final['Order'] == 'PUT') & (df_final['Low'] < df_final['PRED_SAME_DATE']))
    )
    df_final['Precision'] = np.where(conditions, 1, 0)
    
    return df_final

def calculate_pnl(df_final):
    """
    Calculate Profit and Loss (PnL) based on orders and precision.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing 'Order', 'Precision', 'PRED_SAME_DATE', 'Open', 'Close' columns.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'PnL' column.
    """
    required_columns = ['Order', 'Precision', 'PRED_SAME_DATE', 'Open', 'Close']
    for col in required_columns:
        if col not in df_final.columns:
            raise KeyError(f"Column '{col}' is missing from df_final.")
    
    # Vectorized calculation using numpy's select
    conditions = [
        (df_final['Precision'] == 1) & (df_final['Order'] == 'CALL'),
        (df_final['Precision'] == 0) & (df_final['Order'] == 'CALL'),
        (df_final['Precision'] == 1) & (df_final['Order'] == 'PUT'),
        (df_final['Precision'] == 0) & (df_final['Order'] == 'PUT')
    ]
    choices = [
        (df_final['PRED_SAME_DATE'] - df_final['Open']) * 85,
        (df_final['Close'] - df_final['Open']) * 85,
        (df_final['Open'] - df_final['PRED_SAME_DATE']) * 85,
        (df_final['Open'] - df_final['Close']) * 85
    ]
    df_final['PnL'] = np.select(conditions, choices, default=0)
    
    return df_final

def handle_low_pnl(df_final, threshold=-70):
    """
    Replace any PnL values lower than the specified threshold with the threshold value.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing 'PnL' column.
    - threshold (float): The minimum PnL value allowed.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with adjusted 'PnL' values.
    """
    if 'PnL' not in df_final.columns:
        raise KeyError("DataFrame must contain 'PnL' column to handle low PnL values.")
    
    df_final['PnL'] = df_final['PnL'].clip(lower=threshold)
    return df_final

def calculate_efficiency(df_final):
    """
    Determine efficiency based on positive PnL and add an 'Efficiency' column.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing 'PnL' column.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'Efficiency' column.
    """
    if 'PnL' not in df_final.columns:
        raise KeyError("DataFrame must contain 'PnL' column to calculate efficiency.")
    
    df_final['Efficiency'] = np.where(df_final['PnL'] > 0, 1, 0)
    return df_final

def load_backtest_results(filepath='BT_RESULTS.csv'):
    """
    Load existing backtest results from a CSV file.
    
    Parameters:
    - filepath (str): Path to the backtest results CSV file.
    
    Returns:
    - pd.DataFrame: DataFrame containing existing backtest results.
    """
    try:
        bt_results = pd.read_csv(filepath)
    except FileNotFoundError:
        bt_results = pd.DataFrame(columns=['Efficiency', 'PnL'])
    return bt_results

def calculate_average_metrics(bt_results):
    """
    Calculate average efficiency and PnL from backtest results.
    
    Parameters:
    - bt_results (pd.DataFrame): DataFrame containing backtest results.
    
    Returns:
    - tuple: (average_efficiency, average_pnl)
    """
    average_efficiency = bt_results['Efficiency'].mean() if not bt_results.empty else 0
    average_pnl = bt_results['PnL'].mean() if not bt_results.empty else 0
    return average_efficiency, average_pnl

def save_backtest_results(bt_results, filepath='BT_RESULTS.csv'):
    """
    Save backtest results to a CSV file.
    
    Parameters:
    - bt_results (pd.DataFrame): DataFrame containing backtest results.
    - filepath (str): Path to save the backtest results CSV file.
    """
    bt_results.to_csv(filepath, index=False)

def perform_backtest(df_final, filepath='BT_RESULTS.csv', pnl_threshold=-70):
    """
    Perform the entire backtesting process: determine orders, calculate precision, PnL, handle low PnL,
    calculate efficiency, load/save backtest results, and compute average metrics.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing prediction and actual data.
    - filepath (str): Path to the backtest results CSV file.
    - pnl_threshold (float): The minimum PnL value allowed.
    
    Returns:
    - tuple: (average_efficiency, average_pnl)
    """
    # Determine Trading Orders
    df_final = determine_trading_orders(df_final)
    
    # Calculate Precision
    df_final = calculate_precision(df_final)
    
    # Calculate PnL
    df_final = calculate_pnl(df_final)
    
    # Handle Low PnL Values
    df_final = handle_low_pnl(df_final, threshold=pnl_threshold)
    
    # Calculate Efficiency
    df_final = calculate_efficiency(df_final)
    
    # Load Existing Backtest Results
    bt_results = load_backtest_results(filepath)
    
    # Append New Results
    new_results = df_final[['Efficiency', 'PnL']].copy()
    bt_results = pd.concat([bt_results, new_results], ignore_index=True)
    
    # Calculate Averages
    average_efficiency, average_pnl = calculate_average_metrics(bt_results)
    
    # Save Updated Backtest Results
    save_backtest_results(bt_results, filepath)
    
    return average_efficiency, average_pnl
