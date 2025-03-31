import os
import sys
import time
import logging
import openmeteo_requests
import pandas as pd
from datetime import datetime, date, timedelta

HIST_URL = "https://archive-api.open-meteo.com/v1/archive"

def batch_fetch_historical(latitudes, longitudes, start_date, end_date, batch_size=10):
    """
    分批下載歷史資料，若 API 限制錯誤發生則中斷執行。
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
        except openmeteo_requests.Client.OpenMeteoRequestsError as e:
            logging.error(f"API 限制錯誤發生：{e}。終止執行。")
            sys.exit(1)
        results.extend(responses)
        logging.info("等待60秒")
        time.sleep(60)
    return results

def batch_fetch_forecast(latitudes, longitudes, forecast_days, batch_size=10):
    """
    分批下載預報資料，若 API 限制錯誤發生則中斷執行。
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
        except openmeteo_requests.Client.OpenMeteoRequestsError as e:
            logging.error(f"API 限制錯誤發生：{e}。終止執行。")
            sys.exit(1)
        results.extend(responses)
        logging.info("等待60秒")
        time.sleep(60)
    return results
