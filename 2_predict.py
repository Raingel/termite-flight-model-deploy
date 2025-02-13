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
# 下載歷史氣象資料：利用 ERA5_land（ECMWF Seamless）資料
# =============================================================================

HIST_URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_climate_data(latitude, longitude, year, month, day, one_single_day=False):
    """
    根據指定經緯度與年份下載該年度的【歷史】日氣象資料（ERA5_land），
    回傳一個 dict，各變數皆為 numpy array。
    
    當請求年份為當年時，結束日期會自動設定為昨天。
    """
    start_date = f"{year}-01-01"
    date_obj = datetime(year, month, day)
    formatted_date = date_obj.strftime("%Y-%m-%d")
    today = date.today()
    yesterday = today - timedelta(days=1)
    if year == today.year:
        end_date = yesterday.strftime("%Y-%m-%d")
    else:
        end_date = f"{year}-12-31"
    if one_single_day:
        start_date = formatted_date
        end_date = formatted_date

    params = {
        "latitude": latitude,
        "longitude": longitude,
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
    logging.info(f"Fetching historical weather data for lat={latitude}, lon={longitude}, "
                 f"from {start_date} to {end_date}.")
    client = openmeteo_requests.Client()
    responses = client.weather_api(HIST_URL, params=params)
    response = responses[0]
    daily = response.Daily()
    elevation = response.Elevation()
    
    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["elevation"] = elevation
    daily_data["temperature_2m_max"] = daily.Variables(0).ValuesAsNumpy()
    daily_data["temperature_2m_min"] = daily.Variables(1).ValuesAsNumpy()
    daily_data["temperature_2m_mean"] = daily.Variables(2).ValuesAsNumpy()
    daily_data["apparent_temperature_max"] = daily.Variables(3).ValuesAsNumpy()
    daily_data["apparent_temperature_min"] = daily.Variables(4).ValuesAsNumpy()
    daily_data["apparent_temperature_mean"] = daily.Variables(5).ValuesAsNumpy()
    daily_data["daylight_duration"] = daily.Variables(8).ValuesAsNumpy()
    daily_data["sunshine_duration"] = daily.Variables(9).ValuesAsNumpy()
    daily_data["precipitation_sum"] = daily.Variables(10).ValuesAsNumpy()
    daily_data["rain_sum"] = daily.Variables(11).ValuesAsNumpy()
    daily_data["precipitation_hours"] = daily.Variables(12).ValuesAsNumpy()
    daily_data["wind_speed_10m_max"] = daily.Variables(13).ValuesAsNumpy()
    daily_data["wind_gusts_10m_max"] = daily.Variables(14).ValuesAsNumpy()
    daily_data["wind_direction_10m_dominant"] = daily.Variables(15).ValuesAsNumpy()
    daily_data["shortwave_radiation_sum"] = daily.Variables(16).ValuesAsNumpy()
    daily_data["et0_fao_evapotranspiration"] = daily.Variables(17).ValuesAsNumpy()
    
    return daily_data

# =============================================================================
# 下載預報氣象資料：利用 ECMWF IFS 預報資料
# =============================================================================

def fetch_forecast_data(latitude, longitude, forecast_days=16):
    import requests_cache
    from retry_requests import retry
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    forecast_client = openmeteo_requests.Client(session=retry_session)
    fc_url = "https://api.open-meteo.com/v1/forecast"
    daily_vars = [
        "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
        "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
        "daylight_duration", "sunshine_duration", "precipitation_sum", "rain_sum",
        "precipitation_hours", "wind_speed_10m_max", "wind_gusts_10m_max",
        "wind_direction_10m_dominant", "shortwave_radiation_sum", "et0_fao_evapotranspiration"
    ]
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": daily_vars,
        "timezone": "Asia/Singapore",
        "past_days": 10,
        "forecast_days": forecast_days,
        "models": "best_match"
    }
    logging.info(f"Fetching forecast weather data for lat={latitude}, lon={longitude}, forecast for {forecast_days} days.")
    responses = forecast_client.weather_api(fc_url, params=params)
    response = responses[0]
    daily = response.Daily()
    forecast_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    forecast_data["temperature_2m_max"] = daily.Variables(0).ValuesAsNumpy()
    forecast_data["temperature_2m_min"] = daily.Variables(1).ValuesAsNumpy()
    forecast_data["temperature_2m_mean"] = daily.Variables(2).ValuesAsNumpy()
    forecast_data["apparent_temperature_max"] = daily.Variables(3).ValuesAsNumpy()
    forecast_data["apparent_temperature_min"] = daily.Variables(4).ValuesAsNumpy()
    forecast_data["apparent_temperature_mean"] = daily.Variables(5).ValuesAsNumpy()
    forecast_data["daylight_duration"] = daily.Variables(6).ValuesAsNumpy()
    forecast_data["sunshine_duration"] = daily.Variables(7).ValuesAsNumpy()
    forecast_data["precipitation_sum"] = daily.Variables(8).ValuesAsNumpy()
    forecast_data["rain_sum"] = daily.Variables(9).ValuesAsNumpy()
    forecast_data["precipitation_hours"] = daily.Variables(10).ValuesAsNumpy()
    forecast_data["wind_speed_10m_max"] = daily.Variables(11).ValuesAsNumpy()
    forecast_data["wind_gusts_10m_max"] = daily.Variables(12).ValuesAsNumpy()
    forecast_data["wind_direction_10m_dominant"] = daily.Variables(13).ValuesAsNumpy()
    forecast_data["shortwave_radiation_sum"] = daily.Variables(14).ValuesAsNumpy()
    forecast_data["et0_fao_evapotranspiration"] = daily.Variables(15).ValuesAsNumpy()
    return forecast_data

# =============================================================================
# 合併氣象資料 for 一個網格點（當年資料＋預報）
# =============================================================================
def download_weather_data_for_grid(lat, lon, forecast_days=16):
    logging.info(f"開始下載網格點資料：lat={lat}, lon={lon}")
    current_year = datetime.now().year
    hist_data = fetch_climate_data(lat, lon, current_year, 1, 1, one_single_day=False)
    hist_dates = pd.to_datetime(hist_data["date"]).tz_localize(None)
    exec_date = pd.Timestamp.now().normalize()
    hist_mask = hist_dates < exec_date
    hist_dict = {}
    for var in hist_data:
        if var == "date":
            continue
        val = hist_data[var]
        if np.isscalar(val):
            hist_dict[var] = np.full(np.sum(hist_mask), val)
        else:
            hist_dict[var] = np.array(val)[hist_mask]
    hist_df = pd.DataFrame(hist_dict)
    hist_df["date"] = hist_dates[hist_mask]
    hist_df = hist_df.dropna()
    hist_df.sort_values("date", inplace=True)
    logging.info(f"歷史資料下載完成，資料筆數：{len(hist_df)}")
    
    forecast_data = fetch_forecast_data(lat, lon, forecast_days=forecast_days)
    fc_dates = pd.to_datetime(forecast_data["date"]).tz_localize(None)
    fc_dict = {}
    for var in forecast_data:
        if var == "date":
            continue
        fc_dict[var] = np.array(forecast_data[var])
    fc_df = pd.DataFrame(fc_dict)
    fc_df["date"] = fc_dates
    fc_df.sort_values("date", inplace=True)
    logging.info(f"預報資料下載完成，資料筆數：{len(fc_df)}")
    
    combined_df = pd.concat([hist_df, fc_df], ignore_index=True)
    combined_df.drop_duplicates(subset="date", keep="first", inplace=True)
    combined_df.sort_values("date", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df["elevation"] = combined_df["elevation"].fillna(combined_df["elevation"].iloc[0])
    
    start_date = pd.Timestamp(f"{current_year}-01-01")
    end_date = exec_date + pd.Timedelta(days=forecast_days)
    combined_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= end_date)]
    combined_df.reset_index(drop=True, inplace=True)
    
    cum_vars = ["temperature_2m_mean", "apparent_temperature_mean", "daylight_duration",
                "sunshine_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
    for var in cum_vars:
        combined_df[f"cumulative_{var}"] = combined_df[var].cumsum()
    
    combined_df["day"] = combined_df["date"].dt.dayofyear
    combined_df["latitude"] = lat
    combined_df["longitude"] = lon
    
    logging.info(f"網格點資料合併完成，總筆數：{len(combined_df)}")
    return combined_df

# =============================================================================
# 更新所有網格點的氣象資料（當年，結合預報）存成 CSV 至 ./weather_data_tmp/
# =============================================================================
def update_weather_data(forecast_days=16):
    grid_path = "./1_grid_points/taiwan_grid.csv"
    output_dir = "./weather_data_tmp"
    os.makedirs(output_dir, exist_ok=True)
    
    grid_df = pd.read_csv(grid_path)
    logging.info(f"讀取網格點資料，共 {len(grid_df)} 筆。")
    for index, row in grid_df.iterrows():
        lon = row["lon"]
        lat = row["lat"]
        file_name = f"lat_{lat}_lon_{lon}.csv"
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            mod_time = os.path.getmtime(file_path)
            if time.time() - mod_time < 24 * 3600:
                logging.info(f"檔案 {file_name} 在 24 小時內已更新，跳過下載。")
                continue
        try:
            logging.info(f"下載網格點：lat={lat}, lon={lon}")
            df = download_weather_data_for_grid(lat, lon, forecast_days=forecast_days)
            df.to_csv(file_path, index=False)
            logging.info(f"儲存網格點資料 {file_name} 至 {file_path}")
            logging.info("等待 5 秒鐘...")
            time.sleep(5)
        except Exception as e:
            logging.error(f"下載網格點 lat={lat}, lon={lon} 失敗：{e}")

# =============================================================================
# 新增功能：更新指定歷史年度的氣象資料（僅歷史，不結合預報）
# =============================================================================
def update_historical_weather_data(target_year, forecast_days=0):
    """
    下載指定年度（例如 2024 年）的歷史氣象資料，不結合預報，
    並存成 CSV 檔到 ./weather_data_historical/{target_year}/。
    若檔案在過去 24 小時內已更新，則跳過下載。
    """
    grid_path = "./1_grid_points/taiwan_grid.csv"
    output_dir = f"./weather_data_historical/{target_year}"
    os.makedirs(output_dir, exist_ok=True)
    
    grid_df = pd.read_csv(grid_path)
    logging.info(f"[{target_year}] 讀取網格點資料，共 {len(grid_df)} 筆。")
    for index, row in grid_df.iterrows():
        lon = row["lon"]
        lat = row["lat"]
        file_name = f"lat_{lat}_lon_{lon}.csv"
        file_path = os.path.join(output_dir, file_name)
        if os.path.exists(file_path):
            mod_time = os.path.getmtime(file_path)
            # 這裡假設 24 小時內更新為最新
            if time.time() - mod_time < 24 * 3600:
                logging.info(f"[{target_year}] 檔案 {file_name} 在 24 小時內已更新，跳過下載。")
                continue
        try:
            logging.info(f"[{target_year}] 下載網格點：lat={lat}, lon={lon}")
            hist_data = fetch_climate_data(lat, lon, target_year, 1, 1, one_single_day=False)
            dates = pd.to_datetime(hist_data["date"]).tz_localize(None)
            mask = (dates >= pd.Timestamp(f"{target_year}-01-01")) & (dates <= pd.Timestamp(f"{target_year}-12-31"))
            hist_dict = {}
            for var in hist_data:
                if var == "date":
                    continue
                val = hist_data[var]
                if np.isscalar(val):
                    hist_dict[var] = np.full(np.sum(mask), val)
                else:
                    hist_dict[var] = np.array(val)[mask]
            hist_df = pd.DataFrame(hist_dict)
            hist_df["date"] = dates[mask]
            hist_df = hist_df.dropna().sort_values("date").reset_index(drop=True)
            cum_vars = ["temperature_2m_mean", "apparent_temperature_mean", "daylight_duration",
                        "sunshine_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
                        "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
            for var in cum_vars:
                hist_df[f"cumulative_{var}"] = hist_df[var].cumsum()
            hist_df["day"] = pd.to_datetime(hist_df["date"]).dt.dayofyear
            hist_df["latitude"] = lat
            hist_df["longitude"] = lon
            hist_df.to_csv(file_path, index=False)
            logging.info(f"[{target_year}] 儲存歷史氣象資料 {file_name} 至 {file_path}")
            logging.info("等待 5 秒鐘...")
            time.sleep(5)
        except Exception as e:
            logging.error(f"[{target_year}] 下載網格點 lat={lat}, lon={lon} 失敗：{e}")

# =============================================================================
# 預報功能：從指定資料夾載入氣象資料並依日期預測
# =============================================================================
def run_forecast_from_weather_data(input_folder="./weather_data_tmp"):
    logging.info("開始讀取預先訓練的模型...")
    model_files = glob.glob(os.path.join("models", "*_model.pkl"))
    models = {}
    for mf in model_files:
        model_name = os.path.splitext(os.path.basename(mf))[0]
        try:
            with open(mf, "rb") as f:
                models[model_name] = pickle.load(f)
            logging.info(f"成功讀取模型：{model_name}")
        except Exception as e:
            logging.error(f"讀取模型 {mf} 時發生錯誤: {e}")

    expected_daily_keys = ["elevation", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
                           "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean",
                           "daylight_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
                           "wind_speed_10m_max", "wind_gusts_10m_max", "shortwave_radiation_sum",
                           "et0_fao_evapotranspiration", "latitude", "longitude", "day"]
    cum_vars = ["temperature_2m_mean", "apparent_temperature_mean", "daylight_duration",
                "sunshine_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
    expected_cum_keys = [f"cumulative_{var}" for var in cum_vars]

    weather_files = glob.glob(os.path.join(input_folder, "*.csv"))
    logging.info(f"找到 {len(weather_files)} 個氣象資料檔案於 {input_folder}。")
    all_results = []
    for wf in weather_files:
        try:
            df = pd.read_csv(wf)
            logging.info(f"處理檔案：{wf}，共 {len(df)} 筆資料。")
        except Exception as e:
            logging.error(f"讀取檔案 {wf} 失敗：{e}")
            continue
        base = os.path.basename(wf)
        try:
            parts = base.replace(".csv", "").split("_")
            lat_val = float(parts[1])
            lon_val = float(parts[3])
        except Exception as e:
            logging.error(f"解析檔名 {base} 失敗：{e}")
            continue

        for idx, row in df.iterrows():
            feature_dict = {k: row[k] for k in expected_daily_keys if k in row}
            for k in expected_cum_keys:
                if k in row:
                    feature_dict[k] = round(row[k], 4)
            X = pd.DataFrame([feature_dict])
            individual_preds = {}
            cf_preds = []
            cg_preds = []
            ensemble_cf = 0.0
            ensemble_cg = 0.0
            model_weight = {"cf_glm_model":0.10,
                            "cf_lda_model":0.15,
                            "cf_nn_model":0.10,
                            "cf_rf_model":0.10,
                            "cf_svm_model":0.55,
                            "cg_glm_model":0.10,
                            "cg_lda_model":0.10,
                            "cg_nn_model":0.10,
                            "cg_rf_model":0.55,
                            "cg_svm_model":0.15}
            for model_name, model in models.items():
                if model_name not in model_weight:
                    logging.error(f"模型 {model_name} 未指定權重，將跳過此模型。")
                    continue
                try {
                    proba = model.predict_proba(X)[:, 1][0]
                    individual_preds[model_name] = round(proba, 3)
                    if (model_name.startsWith("cf_")) {
                        cf_preds.append(proba)
                        ensemble_cf += proba * model_weight[model_name]
                    } else if (model_name.startsWith("cg_")) {
                        cg_preds.append(proba)
                        ensemble_cg += proba * model_weight[model_name]
                    }
                } catch (e) {
                    logging.error(f"模型 {model_name} 在 {row['date']} 預測失敗：{e}")
                }
            ensemble_cf = float(np.mean(cf_preds)) if cf_preds else None
            ensemble_cg = float(np.mean(cg_preds)) if cg_preds else None
            interaction_score = ensemble_cf * ensemble_cg if (ensemble_cf and ensemble_cf >= 0.5 and ensemble_cg and ensemble_cg >= 0.5) else 0.0
            result = {
                "date": row["date"],
                "individual_predictions": individual_preds,
                "ensemble": {
                    "cf": ensemble_cf,
                    "cg": ensemble_cg,
                    "interaction_score": interaction_score
                },
                "features": {"latitude": lat_val, "longitude": lon_val}
            }
            all_results.append(result)

    logging.info(f"共預測出 {len(all_results)} 筆結果。")
    forecasts_by_date = {}
    for res in all_results:
        ds = pd.to_datetime(res["date"]).strftime("%Y-%m-%d")
        if ds not in forecasts_by_date:
            forecasts_by_date[ds] = []
        forecasts_by_date[ds].append(res)

    output_dir = "./2_predictions"
    os.makedirs(output_dir, exist_ok=True)
    for ds, results_list in forecasts_by_date.items():
        out_path = os.path.join(output_dir, f"{ds}.json")
        for item in results_list:
            if isinstance(item.get("date"), (pd.Timestamp, datetime)):
                item["date"] = pd.to_datetime(item["date"]).strftime("%Y-%m-%d")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        logging.info(f"儲存預報結果 {ds} 至 {out_path}")

# =============================================================================
# 主入口：使用 argparse 決定要執行哪個模式
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Weather Data and Forecast Runner")
    parser.add_argument("mode", choices=["update", "forecast", "update_historical"], help="Mode to run: update (update current year's weather data), forecast (run forecast using downloaded weather data), update_historical (update historical weather data for a specified year)")
    parser.add_argument("--year", type=int, help="Target year for historical weather data (required for update_historical mode)")
    parser.add_argument("--forecast_days", type=int, default=16, help="Number of forecast days (default: 16)")
    args = parser.parse_args()

    if args.mode == "update":
        logging.info("開始更新當年氣象資料...")
        update_weather_data(forecast_days=args.forecast_days)
        logging.info("當年氣象資料更新完成。")
    elif args.mode == "forecast":
        logging.info("開始依當年氣象資料進行預報...")
        run_forecast_from_weather_data(input_folder="./weather_data_tmp")
        logging.info("當年預報結果已儲存至 ./2_predictions/。")
    elif args.mode == "update_historical":
        if args.year is None:
            logging.error("請提供目標年份 --year (例如 --year 2024)")
            return
        logging.info(f"開始更新 {args.year} 年歷史氣象資料...")
        update_historical_weather_data(args.year, forecast_days=0)
        logging.info(f"{args.year} 年歷史氣象資料更新完成。")
    else:
        logging.error("未知的模式")
        return

if __name__ == "__main__":
    main()
