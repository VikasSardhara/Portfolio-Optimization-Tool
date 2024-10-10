                           # Portfolio Optimization Tool

This is a Python-based portfolio optimization tool that helps users allocate investments into various asset classes such as stocks, bonds, gold, and real estate. It utilizes **Modern Portfolio Theory (MPT)** to optimize for minimal risk while meeting the desired return.

## Features
- Fetches historical data for **S&P 500**, **Bonds (TLT)**, and **Gold (GLD)** using `yfinance`.
- Allows for **user-defined risk tolerance** and **desired return** (7.75% to 9%).
- Option to include **real estate investment**.
- Performs portfolio optimization using **scipy.optimize.minimize**.
- Displays optimized asset allocation with expected return and risk.
- Optional visualization of the **Efficient Frontier**.

## Requirements

- Python 3.x
- Required Libraries: `yfinance`, `numpy`, `pandas`, `scipy`, `matplotlib`

Install the required libraries:
```bash
pip install yfinance numpy pandas scipy matplotlib
