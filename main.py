import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function to fetch historical data for S&P 500, Bonds, and Gold
def get_historical_data():
    print("Fetching historical data...")
    
    try:
        # S&P 500 Total Return Index
        sp500_tr = yf.download("^SP500TR", start="2015-01-01", end="2023-01-01", progress=False)
        sp500_returns = sp500_tr['Adj Close'].pct_change().mean() * 252  # Annualized returns
        print("S&P 500 Total Return data fetched.")

        # Bond ETF (TLT - 20+ Year Treasury Bond ETF)
        bond_data = yf.download("TLT", start="2015-01-01", end="2023-01-01", progress=False)
        bond_returns = bond_data['Adj Close'].pct_change().mean() * 252  # Annualized returns
        print("Bond data fetched.")

        # Gold ETF (GLD - SPDR Gold Trust)
        gold_data = yf.download("GLD", start="2015-01-01", end="2023-01-01", progress=False)
        gold_returns = gold_data['Adj Close'].pct_change().mean() * 252  # Annualized returns
        print("Gold data fetched.")

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None, None, None, None, None

    return sp500_returns, bond_returns, gold_returns, sp500_tr, bond_data, gold_data

# Function to gather user input on investment preferences
def get_user_inputs():
    while True:
        term = input("Choose investment term (medium or long): ").lower()
        if term in ['medium', 'long']:
            break
        else:
            print("Invalid input. Please enter 'medium' or 'long'.")

    while True:
        try:
            risk_tolerance = float(input("Enter your risk tolerance (1-100%): "))
            if 1 <= risk_tolerance <= 100:
                break
            else:
                print("Please enter a valid percentage between 1 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            desired_return = float(input("Enter the desired return (between 7.75% and 9%): "))
            if 7.75 <= desired_return <= 9.0:
                break
            else:
                print("Please enter a return between 7.75% and 9%.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        invest_real_estate = input("Do you want to invest in real estate? (yes/no): ").lower()
        if invest_real_estate in ['yes', 'no']:
            break
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")

    if invest_real_estate == 'yes':
        real_estate_returns = float(input("Please enter expected real estate return (%): "))
    else:
        real_estate_returns = 0.0  # Exclude real estate

    return term, risk_tolerance, desired_return, real_estate_returns

# Function to calculate portfolio performance (return)
def portfolio_performance(weights, returns):
    return np.dot(weights, returns)

# Optimization function to adjust weights for desired return
def optimize_portfolio(returns, cov_matrix, target_return):
    num_assets = len(returns)
    
    # Initial guess for weights
    initial_weights = np.array([1 / num_assets] * num_assets)
    
    # Constraints: Weights sum to 1 and portfolio return matches target return
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda weights: portfolio_performance(weights, returns) - target_return}  # Target return
    ]
    
    # Bounds for weights: between 0 and 1
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Objective: Minimize portfolio volatility (standard deviation)
    result = minimize(
        lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))),  # Minimize volatility
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed. Please check the input parameters.")

# Function to display return breakdown by asset
def display_return_breakdown(weights, returns, asset_names):
    print("\n--- Return Breakdown by Asset ---")
    for i, (weight, asset_return) in enumerate(zip(weights, returns)):
        contribution = weight * asset_return * 100
        print(f"{asset_names[i]}: {contribution:.2f}% return contribution")

# Main function to run the investment tool
def main():
    try:
        # Fetch historical data
        sp500_return, bond_return, gold_return, sp500_data, bond_data, gold_data = get_historical_data()
        
        # Get user inputs
        term, risk_tolerance, desired_return, real_estate_return = get_user_inputs()
        
        # Combine asset returns
        returns = np.array([sp500_return, bond_return, gold_return])
        asset_names = ["S&P 500", "Bonds", "Gold"]
        if real_estate_return != 0:
            returns = np.append(returns, real_estate_return)
            asset_names.append("Real Estate")
        
        # Calculate covariance matrix for accurate risk calculation
        price_data = pd.concat([sp500_data['Adj Close'], bond_data['Adj Close'], gold_data['Adj Close']], axis=1)
        price_data.columns = ['S&P 500', 'Bonds', 'Gold']
        price_data = price_data.dropna()  # Drop missing data points
        daily_returns = price_data.pct_change().dropna()
        cov_matrix = daily_returns.cov() * 252  # Annualized covariance matrix
        
        # Optimize the portfolio to meet the desired return
        target_return_decimal = desired_return / 100
        optimized_weights = optimize_portfolio(returns, cov_matrix, target_return_decimal)
    
        # Display the optimized allocation
        print("\n--- Optimized Portfolio Allocation ---")
        for name, weight in zip(asset_names, optimized_weights):
            print(f"{name}: {weight * 100:.2f}%")
    
        # Display the portfolio's expected return
        portfolio_return = portfolio_performance(optimized_weights, returns)
        print(f"\nExpected Portfolio Return: {portfolio_return * 100:.2f}%")
    
        # Display the return breakdown by asset
        display_return_breakdown(optimized_weights, returns, asset_names)
    
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the program
if __name__ == "__main__":
    main()
