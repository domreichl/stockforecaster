import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf


def update_data(df: pd.DataFrame, ISIN: str) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.sort_values("Date", ascending=False, inplace=True)
    new = download_data(ISIN, start=df["Date"].iloc[2])
    updated = pd.concat([df, new])
    updated.drop_duplicates("Date", keep="last", inplace=True)
    updated.sort_values("Date", ascending=False, inplace=True)
    return updated


def download_data(ISIN: str, start: dt.datetime) -> pd.DataFrame:
    df = yf.download(ISIN, start=start, end=dt.datetime.now()).reset_index()
    assert len(df[df["Date"].duplicated()]) == 0
    df.drop(columns=["Adj Close"], inplace=True)
    df = impute_missing_rows(df)
    df.sort_values("Date", ascending=False, inplace=True)
    return df


def impute_missing_rows(df: pd.DataFrame) -> pd.DataFrame:
    expected = pd.date_range(df["Date"].min(), df["Date"].max(), freq="B").date
    actual = pd.to_datetime(df["Date"].unique()).date
    missing = list(set(expected).difference(set(actual)))
    print(f"Filling in {len(missing)} missing rows")
    imputing_df = pd.DataFrame({"Date": pd.to_datetime(missing)})
    for col in df.columns:
        if col != "Date":
            imputing_df[col] = np.nan
    df = pd.concat([df, imputing_df])
    df.sort_values("Date", inplace=True)
    for col in df.columns:
        if col != "Date":
            df[col] = df[col].ffill()
    df["Date"] = df["Date"].dt.date
    return df
