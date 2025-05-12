import pandas as pd
import numpy as np
from datetime import datetime


def convert_to_daily(input_path, output_path):
    """
    Chuyển đổi dữ liệu từ interval 1 phút sang 1 ngày

    Args:
        input_path: đường dẫn đến file CSV gốc (interval 1 phút)
        output_path: đường dẫn để lưu file CSV mới (interval 1 ngày)
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

    # Resample sang daily
    print("Đang chuyển đổi sang dữ liệu daily...")
    daily_df = pd.DataFrame()

    # Open: giá đầu tiên của ngày
    daily_df["Open"] = df.resample("D", on="Date")["Open"].first()

    # High: giá cao nhất của ngày
    daily_df["High"] = df.resample("D", on="Date")["High"].max()

    # Low: giá thấp nhất của ngày
    daily_df["Low"] = df.resample("D", on="Date")["Low"].min()

    # Close: giá cuối cùng của ngày
    daily_df["Close"] = df.resample("D", on="Date")["Close"].last()

    # Volume: tổng volume của ngày
    daily_df["Volume"] = df.resample("D", on="Date")["Volume"].sum()

    # Reset index để lấy Date làm cột
    daily_df = daily_df.reset_index()

    # Chuyển Date về timestamp
    daily_df["timestamp"] = daily_df["Date"].astype(np.int64) // 10**9

    # Chọn và sắp xếp các cột
    daily_df = daily_df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]

    # Lưu file
    print(f"Đang lưu dữ liệu daily vào {output_path}...")
    daily_df.to_csv(output_path, index=False)

    print("Hoàn thành!")
    print(f"Số lượng dữ liệu gốc: {len(df)} records")
    print(f"Số lượng dữ liệu daily: {len(daily_df)} records")

    return daily_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chuyển đổi dữ liệu từ interval 1 phút sang 1 ngày"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Đường dẫn đến file CSV gốc"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Đường dẫn để lưu file CSV mới"
    )

    args = parser.parse_args()

    convert_to_daily(args.input, args.output)
