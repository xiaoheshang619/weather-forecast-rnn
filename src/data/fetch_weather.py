"""
从 Open-Meteo Archive API 获取历史天气数据。
无需 API Key，免费使用。
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
LAT = 31.7659
LON = 117.1855
START_DATE = "2023-05-13"
END_DATE = "2026-05-13"
OUTPUT_PATH = Path("data/raw/weather_hefei.csv")

# Open-Meteo Archive API 端点
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# 请求的每日天气变量
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


def build_params() -> dict:
    """构造 API 请求参数"""
    return {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Shanghai",
    }


def fetch_weather() -> pd.DataFrame:
    """拉取数据并返回清洗后的 DataFrame"""
    print(f"[*] 正在请求 Open-Meteo Archive API ...")
    print(f"    坐标: ({LAT}, {LON})")
    print(f"    时间: {START_DATE} ~ {END_DATE}")

    resp = requests.get(ARCHIVE_URL, params=build_params(), timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if "daily" not in data:
        raise RuntimeError(f"API 返回异常: {data}")

    # 解析为 DataFrame
    records = data["daily"]
    df = pd.DataFrame(records)

    # date 列转为 datetime
    df["date"] = pd.to_datetime(df["time"])
    df.drop(columns=["time"], inplace=True)

    # 调整列顺序，date 放第一列
    cols = ["date"] + [c for c in df.columns if c != "date"]
    df = df[cols]

    # 基本清洗：删去全空行
    df.dropna(how="all", subset=DAILY_VARS, inplace=True)

    return df


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_weather()

    # 保存原始数据
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[√] 数据已保存至 {OUTPUT_PATH}")
    print(f"    形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"    日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(f"    列名: {list(df.columns)}")

    # 简要统计
    print(f"\n── 缺失值统计 ──")
    for col in DAILY_VARS:
        n_miss = df[col].isna().sum()
        if n_miss > 0:
            print(f"  {col}: {n_miss} 个缺失")
    print("  完成。")


if __name__ == "__main__":
    main()
