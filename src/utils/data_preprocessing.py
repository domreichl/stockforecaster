import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta.trend import macd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

from utils.config import Config
from utils.data_classes import MultivariateTimeSeries
from utils.file_handling import DataHandler


def preprocess_data(
    look_back_window: int = None,
    split_type: str = "validation",
) -> MultivariateTimeSeries:
    cfg = Config()
    dh = DataHandler()
    df = dh.load_csv_data(cfg.ISIN)
    df, dates = trim_and_sort_data(df, cfg.holding_days)
    index_df = dh.load_csv_data(cfg.index)
    index_df, _ = trim_and_sort_data(index_df, cfg.holding_days)
    x, y = process_features(
        df,
        index_df["Close"],
        cfg.features,
        look_back_window or cfg.look_back_window,
        cfg.holding_days,
        cfg.buy_threshold,
    )
    x_train, y_train, x_test, y_test = split_sets(
        x, y, cfg.train_days, cfg.eval_days, split_type
    )
    x_train, x_test = normalize(x_train, x_test)
    print("Shapes:")
    print(" x:", x_train.shape, x_test.shape)
    print(" y:", y_train.shape, y_test.shape)
    return MultivariateTimeSeries(dates, x_train, y_train, x_test, y_test)


def trim_and_sort_data(df: pd.DataFrame, holding_days: int) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    n_units = df["Date"].nunique() - df["Date"].nunique() % holding_days
    dates = sorted(list(df["Date"].unique()))[-n_units:]
    df = df[df["Date"].isin(dates)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df, dates


def process_features(
    df: pd.DataFrame,
    index_close: pd.Series,
    features: list,
    look_back_window: int,
    holding_days: int,
    buy_threshold: float,
) -> tuple[np.array]:
    x = []
    T = len(df) - look_back_window - holding_days
    for feature_name in features:
        match feature_name:
            case "RawClose":
                feature = df["Close"]
            case "LogReturn":
                feature = compute_log_returns(df["Close"])
            case "Sign":
                feature = compute_binary_signs(df["Close"], buy_threshold)
            case "Volume":
                feature = df["Volume"]
            case "DayOfWeek":
                feature = np.sin(2 * np.pi * pd.to_datetime(df["Date"]).dt.day_of_week)
            case "HolidaysAT":
                holidays_at = holidays.AT()
                feature = pd.Series([d in holidays_at for d in df["Date"]])
            case "RPR":  # Relative Price Range
                feature = 2 * (df["High"] - df["Low"]) / (df["High"] + df["Low"])
            case "RSI":  # Relative Strength Index
                feature = RSIIndicator(df["Close"]).rsi()
            case "StochasticOscillator":
                feature = StochasticOscillator(
                    df["High"], df["Low"], df["Close"]
                ).stoch_signal()
            case "MACD":  # Moving Average Convergence/Divergence
                feature = macd(df["Close"])
            case "EMA5":
                feature = df["Close"].ewm(span=5).mean()
            case "EMA20":
                feature = df["Close"].ewm(span=20).mean()
            case "BollingerBandHigh":
                feature = BollingerBands(df["Close"]).bollinger_hband_indicator()
            case "BollingerBandLow":
                feature = BollingerBands(df["Close"]).bollinger_lband_indicator()
            case "RawIndex":
                feature = index_close
            case "LogReturnIndex":
                feature = compute_log_returns(index_close)
            case "SignIndex":
                feature = compute_binary_signs(index_close, buy_threshold)
            case _:
                raise Exception(f"Feature {feature_name} is not implemented.")
        x.append(build_lagged_feature(feature, T, look_back_window))
    x = np.stack(x, 2)  # T x WindowSize x Features
    y = np.array(
        [
            np.int32(
                np.mean(
                    compute_returns(df["Close"])[
                        t + look_back_window : t + look_back_window + holding_days
                    ]
                )
                > buy_threshold
            )
            for t in range(T)
        ],
    )
    return x, y


def compute_returns(prices: pd.Series) -> np.array:
    return np.array(prices / prices.shift(1).fillna(0.0))


def compute_log_returns(prices: pd.Series) -> np.array:
    return np.log(compute_returns(prices))


def compute_binary_signs(prices: pd.Series, buy_threshold: float) -> np.array:
    return np.float32(compute_returns(prices) > buy_threshold)


def build_lagged_feature(feature: pd.Series, T: int, look_back_window: int) -> np.array:
    return np.stack([np.array(feature)[t : t + look_back_window] for t in range(T)], 0)


def split_sets(
    x: np.array, y: np.array, train_days: int, eval_days: int, split_type: str
) -> None:
    if split_type == "validation":
        x_train, y_train = x[: -eval_days * 2], y[: -eval_days * 2]
        x_test, y_test = (
            x[-eval_days * 2 : -eval_days],
            y[-eval_days * 2 : -eval_days],
        )
    elif split_type == "test":
        x_train, y_train = x[:-eval_days], y[:-eval_days]
        x_test, y_test = x[-eval_days:], y[-eval_days:]
    elif split_type == "forecast":
        x_train, y_train = x, y
        x_test, y_test = np.array([]), np.array([])
    return x_train[-train_days:], y_train[-train_days:], x_test, y_test


def normalize(x_train, x_test) -> tuple:
    train, test = [], []
    for feat in range(x_train.shape[2]):
        scaler = MinMaxScaler()
        train.append(
            scaler.fit_transform(x_train[:, :, feat].reshape(-1, 1)).reshape(
                x_train[:, :, feat].shape
            )
        )
        test.append(
            scaler.transform(x_test[:, :, feat].reshape(-1, 1)).reshape(
                x_test[:, :, feat].shape
            )
        )
    return np.stack(train, 2), np.stack(test, 2)
