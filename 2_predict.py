# 2_predict.py

import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import openmeteo_requests
import json
import time
import logging
import argparse

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# 批次下載歷史氣象資料：利用 ERA5_land（ECMWF Seamless）
# =============================================================================
HIST_URL = "https://archive-api.open-meteo.com/v1/archive"

def batch_fetch_historical(latitudes, longitudes, start_date, end_date, batch_size=10):
    """
    依據提供的經緯度列表，分批下載歷史資料。
    遇到 API exception 時就跳出並回傳已下載的結果。
    """
    results = []
    client = openmeteo_requests.Client()
    for i in range(0, len(latitudes), batch_size):
        chunk_lats = list(latitudes[i:i+batch_size])
        chunk_lons = list(longitudes[i:i+batch_size])
        params = {
            "latitude": chunk_lats,
            "longitude": chunk_lons,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
                "sunrise", "sunset", "daylight_duration", "sunshine_duration", "precipitation_sum",
                "rain_sum", "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
                "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
            ],
            "timezone": "Asia/Singapore",
            "models": "era5_seamless"
        }
        logging.info(f"批次下載歷史資料：處理第 {i+1} 到 {i+len(chunk_lats)} 個點")
        try:
            responses = client.weather_api(HIST_URL, params=params)
            results.extend(responses)
        except Exception as e:
            logging.error(f"歷史資料下載失敗（第 {i+1} 到 {i+len(chunk_lats)} 個點）：{e}")
            break
        logging.info("等待60秒")
        time.sleep(60)
    return results

# =============================================================================
# 批次下載預報氣象資料：利用 ECMWF IFS
# =============================================================================
def batch_fetch_forecast(latitudes, longitudes, forecast_days, batch_size=10):
    """
    依據提供的經緯度列表，分批下載預報資料。
    遇到 API exception 時就跳出並回傳已下載的結果。
    """
    results = []
    client = openmeteo_requests.Client()
    for i in range(0, len(latitudes), batch_size):
        chunk_lats = list(latitudes[i:i+batch_size])
        chunk_lons = list(longitudes[i:i+batch_size])
        params = {
            "latitude": chunk_lats,
            "longitude": chunk_lons,
            "daily": [
                "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
                "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum",
                "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
                "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
            ],
            "timezone": "Asia/Singapore",
            "past_days": 10,
            "forecast_days": forecast_days,
            "models": "best_match"
        }
        logging.info(f"批次下載預報資料：處理第 {i+1} 到 {i+len(chunk_lats)} 個點")
        try:
            responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
            results.extend(responses)
        except Exception as e:
            logging.error(f"預報資料下載失敗（第 {i+1} 到 {i+len(chunk_lats)} 個點）：{e}")
            break
        logging.info("等待60秒")
        time.sleep(60)
    return results

# 以下 process_historical_response, process_forecast_response, merge_weather_data 等函數保持不變
# …（略）…

# =============================================================================
# 更新所有網格點的當年氣象資料（結合預報），以批次方式下載
# 遇到任何下載錯誤都會先處理已拿到的資料，並正常結束
# =============================================================================
def update_weather_data(forecast_days=16, batch_size=10):
    grid_path = "./1_grid_points/taiwan_grid.csv"
    output_dir = "./weather_data_tmp"
    os.makedirs(output_dir, exist_ok=True)
    
    grid_df = pd.read_csv(grid_path)
    logging.info(f"讀取網格點資料，共 {len(grid_df)} 筆。")
    points_to_update = []
    meta_files = {}
    for _, row in grid_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        meta_file = os.path.join(output_dir, f"lat_{lat}_lon_{lon}.meta")
        file_name = f"lat_{lat}_lon_{lon}.csv"
        if os.path.exists(meta_file):
            last_update = float(open(meta_file).read().strip())
            if time.time() - last_update < 20 * 3600:
                logging.info(f"{file_name} 在 20 小時內已更新，跳過。")
                continue
        points_to_update.append((lat, lon))
        meta_files[(lat, lon)] = meta_file

    if not points_to_update:
        logging.info("所有網格點皆在 20 小時內更新，無需下載。")
        return

    lats, lons = zip(*points_to_update)
    current_year = datetime.now().year
    today = date.today()
    if current_year == datetime.now().year:
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = f"{current_year}-12-31"
    start_date = f"{current_year}-01-01"

    # 下載並處理歷史資料（遇錯則跳出，回傳已成功的回應列表）
    hist_responses = batch_fetch_historical(lats, lons, start_date, end_date, batch_size=batch_size)
    hist_data_dict = {}
    for i, response in enumerate(hist_responses):
        try:
            df = process_historical_response(response)
            hist_data_dict[(lats[i], lons[i])] = df
            logging.info(f"歷史資料處理成功：({lats[i]}, {lons[i]})，筆數：{len(df)}")
        except Exception as e:
            logging.error(f"處理歷史資料失敗：({lats[i]}, {lons[i]})，錯誤：{e}")

    # 下載並處理預報資料
    fc_responses = batch_fetch_forecast(lats, lons, forecast_days, batch_size=batch_size)
    fc_data_dict = {}
    for i, response in enumerate(fc_responses):
        try:
            df = process_forecast_response(response)
            fc_data_dict[(lats[i], lons[i])] = df
            logging.info(f"預報資料處理成功：({lats[i]}, {lons[i]})，筆數：{len(df)}")
        except Exception as e:
            logging.error(f"處理預報資料失敗：({lats[i]}, {lons[i]})，錯誤：{e}")

    # 合併 & 存檔（只對已取得歷史資料的點進行合併，即使預報資料不完整也會執行）
    for point in points_to_update:
        lat, lon = point
        if point not in hist_data_dict:
            logging.warning(f"跳過 {point}：無歷史資料可合併")
            continue
        hist_df = hist_data_dict[point]
        fc_df = fc_data_dict.get(point, pd.DataFrame())
        combined_df = merge_weather_data(hist_df, fc_df, current_year, forecast_days)
        file_name = f"lat_{lat}_lon_{lon}.csv"
        file_path = os.path.join(output_dir, file_name)
        combined_df.to_csv(file_path, index=False)
        with open(meta_files[point], "w") as f:
            f.write(str(time.time()))
        logging.info(f"儲存 {file_name} 至 {file_path}")
        time.sleep(5)

# 主程式和 argparse 保持不變
# …（略）…

if __name__ == "__main__":
    main()
