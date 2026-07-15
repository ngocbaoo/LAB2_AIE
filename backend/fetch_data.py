"""
Tai du lieu gia dong cua SPY (ETF theo doi S&P 500) tu Yahoo Finance va luu
vao backend/data/spy.csv. Chi can chay 1 lan (hoac khi muon lam moi du lieu)
-- TradingEnv doc thang tu file CSV nay, khong goi mang trong luc train.

Usage: py backend/fetch_data.py
"""
import os

import yfinance as yf

TICKER = "SPY"
START = "2015-01-01"
END = "2024-12-31"
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "spy.csv")


def main():
    df = yf.download(TICKER, start=START, end=END, progress=False)[["Close"]]
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"Saved {len(df)} rows of {TICKER} close prices to {OUT_PATH}")


if __name__ == "__main__":
    main()
