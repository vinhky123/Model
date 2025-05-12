import pandas as pd
import numpy as np
from datetime import datetime


def add_technical_indicators(df):
    """
    Thêm các chỉ báo kỹ thuật vào DataFrame

    Args:
        df: DataFrame chứa dữ liệu OHLCV
    """
    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Histogram"] = df["MACD"] - df["Signal_Line"]

    # Bollinger Bands
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["BB_Upper"] = df["SMA_20"] + 2 * df["Close"].rolling(window=20).std()
    df["BB_Lower"] = df["SMA_20"] - 2 * df["Close"].rolling(window=20).std()

    # ATR
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR"] = true_range.rolling(14).mean()

    # VWAP
    df["VWAP"] = (
        df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
    ).cumsum() / df["Volume"].cumsum()

    # Momentum
    df["Momentum"] = df["Close"] - df["Close"].shift(10)

    # Rate of Change (ROC)
    df["ROC"] = ((df["Close"] - df["Close"].shift(10)) / df["Close"].shift(10)) * 100

    # Đưa tất cả NaN về 0 cho các chỉ báo kỹ thuật
    indicator_cols = [
        "RSI",
        "MACD",
        "Signal_Line",
        "MACD_Histogram",
        "SMA_20",
        "BB_Upper",
        "BB_Lower",
        "ATR",
        "VWAP",
        "Momentum",
        "ROC",
    ]
    for col in indicator_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    return df


def preprocess_data(input_path, output_path, interval="H"):
    """
    Tiền xử lý dữ liệu: chuyển đổi interval và thêm các chỉ báo kỹ thuật

    Args:
        input_path: đường dẫn đến file CSV gốc
        output_path: đường dẫn để lưu file CSV đã xử lý
        interval: khoảng thời gian ('H' cho hourly, 'D' cho daily)
    """
    # Đọc dữ liệu
    print(f"Đang đọc dữ liệu từ {input_path}...")
    df = pd.read_csv(input_path)

    # Chuyển timestamp sang datetime
    if "Timestamp" in df.columns:
        df["Date"] = pd.to_datetime(df["Timestamp"], unit="s")
    elif "time" in df.columns:
        df["Date"] = pd.to_datetime(df["time"], unit="s")
    else:
        raise ValueError("Không tìm thấy cột timestamp hoặc time trong dữ liệu")

    # Sắp xếp theo thời gian
    df = df.sort_values("Date")

    # Resample sang interval mới
    print(f"Đang chuyển đổi sang dữ liệu {interval}...")
    resampled_df = pd.DataFrame()

    # Open: giá đầu tiên của interval
    resampled_df["Open"] = df.resample(interval, on="Date")["Open"].first()

    # High: giá cao nhất của interval
    resampled_df["High"] = df.resample(interval, on="Date")["High"].max()

    # Low: giá thấp nhất của interval
    resampled_df["Low"] = df.resample(interval, on="Date")["Low"].min()

    # Close: giá cuối cùng của interval
    resampled_df["Close"] = df.resample(interval, on="Date")["Close"].last()

    # Volume: tổng volume của interval
    resampled_df["Volume"] = df.resample(interval, on="Date")["Volume"].sum()

    # Reset index để lấy Date làm cột
    resampled_df = resampled_df.reset_index()

    # Thêm các chỉ báo kỹ thuật
    print("Đang tính toán các chỉ báo kỹ thuật...")
    resampled_df = add_technical_indicators(resampled_df)

    # Chuyển Date về timestamp
    resampled_df["timestamp"] = resampled_df["Date"].astype(np.int64) // 10**9

    # Xóa các cột không cần thiết
    resampled_df = resampled_df.drop(columns=["Date"])

    # Lưu file
    print(f"Đang lưu dữ liệu đã xử lý vào {output_path}...")
    resampled_df.to_csv(output_path, index=False)

    print("Hoàn thành!")
    print(f"Số lượng dữ liệu gốc: {len(df)} records")
    print(f"Số lượng dữ liệu sau khi xử lý: {len(resampled_df)} records")

    return resampled_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Tiền xử lý dữ liệu: chuyển đổi interval và thêm các chỉ báo kỹ thuật"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Đường dẫn đến file CSV gốc"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Đường dẫn để lưu file CSV đã xử lý"
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="H",
        choices=["H", "D"],
        help="Khoảng thời gian (H: hourly, D: daily)",
    )

    args = parser.parse_args()

    preprocess_data(args.input, args.output, args.interval)
