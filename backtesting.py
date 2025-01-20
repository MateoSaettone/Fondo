# backtesting.py

import pandas as pd
import numpy as np

def calculate_precision(df_final):
    """
    Determine if the prediction was correct and add a 'Precision' column.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing prediction and actual data.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'Precision' column.
    """
    df_final['Precision'] = df_final.apply(
        lambda row: 1 if (
            (row['Order'] == 'CALL' and row['High'] > row['PRED_SAME_DATE']) or 
            (row['Order'] == 'PUT' and row['Low'] < row['PRED_SAME_DATE'])
        ) else 0, axis=1
    )
    return df_final

def calculate_pnl(df_final):
    """
    Calculate Profit and Loss (PnL) based on orders and precision.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing 'Order' and 'Precision' columns.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'PnL' column.
    """
    df_final['PnL'] = df_final.apply(
        lambda row: (
            (row['PRED_SAME_DATE'] - row['Open']) * 85 if row['Precision'] == 1 and row['Order'] == 'CALL' else
            (row['Close'] - row['Open']) * 85 if row['Precision'] == 0 and row['Order'] == 'CALL' else
            (row['Open'] - row['PRED_SAME_DATE']) * 85 if row['Precision'] == 1 and row['Order'] == 'PUT' else
            (row['Open'] - row['Close']) * 85 if row['Precision'] == 0 and row['Order'] == 'PUT' else 0
        ), axis=1
    )
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
    df_final['PnL'] = df_final['PnL'].apply(lambda x: threshold if x < threshold else x)
    return df_final

def calculate_efficiency(df_final):
    """
    Determine efficiency based on positive PnL and add an 'Efficiency' column.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing 'PnL' column.
    
    Returns:
    - pd.DataFrame: Updated DataFrame with 'Efficiency' column.
    """
    df_final['Efficiency'] = df_final['PnL'].apply(lambda x: 1 if x > 0 else 0)
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

def perform_backtest(df_final, filepath='BT_RESULTS.csv'):
    """
    Perform the entire backtesting process: calculate precision, PnL, handle low PnL, calculate efficiency,
    load existing results, append new results, calculate averages, save updated results, and return averages.
    
    Parameters:
    - df_final (pd.DataFrame): DataFrame containing prediction and actual data.
    - filepath (str): Path to the backtest results CSV file.
    
    Returns:
    - tuple: (average_efficiency, average_pnl)
    """
    # Calculate Precision
    df_final = calculate_precision(df_final)
    
    # Calculate PnL
    df_final = calculate_pnl(df_final)
    
    # Handle Low PnL Values
    df_final = handle_low_pnl(df_final)
    
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
