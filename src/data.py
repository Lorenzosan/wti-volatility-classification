import yfinance as yf
import pandas as pd
import numpy as np


def extract_prices(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    price_dict: dict[str, pd.Series] = {}

    if (ticker, "Adj Close") in data.columns:
        price_dict[ticker] = data[(ticker, "Adj Close")]
    elif (ticker, "Close") in data.columns:
        price_dict[ticker] = data[(ticker, "Close")]
    else:
        raise KeyError(f"No suitable price column found for {ticker}")

    prices = pd.DataFrame(price_dict).sort_index()
    return prices.dropna()


def download_data(cfg) -> pd.DataFrame:
    raw = yf.download(
        tickers=cfg.data.asset,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
    )
    prices = extract_prices(raw, cfg.data.asset)
    print(f"INFO: successful download for {cfg.data.asset} data")
    return prices



def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    prices = prices.sort_index()

    # Keep only rows where all prices are strictly positive
    valid = (prices > 0).all(axis=1)
    prices = prices.loc[valid]

    returns = np.log(prices / prices.shift(1))
    return returns.dropna()
