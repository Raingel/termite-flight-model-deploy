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
# 批次下載歷史氣象資料：利用 ERA5_land（ECMWF Seamless）資料
# =============================================================================
HIST_URL = "https://archive-api.open-meteo.com/v1/archive"

def batch_fetch_historical(latitudes, longitudes, start_date, end_date, batch_size=10):
    """
    依據提供的經緯度列表，分批下載歷史資料，回傳一個包含所有點回應的列表。
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
        responses = client.weather_api(HIST_URL, params=params)
        results.extend(responses)
        # 等待60秒
        logging.info("等待60秒")
        time.sleep(60)
    return results

# =============================================================================
# 批次下載預報氣象資料：利用 ECMWF IFS 預報資料
# =============================================================================
def batch_fetch_forecast(latitudes, longitudes, forecast_days, batch_size=10):
    """
    依據提供的經緯度列表，分批下載預報資料，回傳一個包含所有點回應的列表。
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
        responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        results.extend(responses)
        # 等待60秒
        logging.info("等待60秒")
        time.sleep(60)
    return results

# =============================================================================
# 處理單一回應，轉成 DataFrame
# =============================================================================
def process_historical_response(response):
    daily = response.Daily()
    daily_data = {"date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    )}
    daily_data["elevation"] = response.Elevation()
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
    df = pd.DataFrame(daily_data)
    return df

def process_forecast_response(response):
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
    df = pd.DataFrame(forecast_data)
    return df

# =============================================================================
# 合併歷史與預報資料，並計算累積特徵 for 一個網格點
# =============================================================================
def merge_weather_data(hist_df, fc_df, current_year, forecast_days):
    exec_date = pd.Timestamp.now().normalize()
    start_date = pd.Timestamp(f"{current_year}-01-01").tz_localize('UTC')
    end_date = (exec_date + pd.Timedelta(days=forecast_days)).tz_localize('UTC')
    hist_df.dropna(inplace=True)
    combined_df = pd.concat([hist_df, fc_df], ignore_index=True)
    combined_df.drop_duplicates(subset="date", keep="first", inplace=True)
    combined_df = combined_df[combined_df["date"] >= start_date]
    combined_df = combined_df[combined_df["date"] <= end_date]
    combined_df.sort_values("date", inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df["elevation"] = combined_df["elevation"].fillna(combined_df["elevation"].iloc[0])
    cum_vars = ["temperature_2m_mean", "apparent_temperature_mean", "daylight_duration",
                "sunshine_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
    for var in cum_vars:
        combined_df[f"cumulative_{var}"] = combined_df[var].cumsum()
    combined_df["day"] = combined_df["date"].dt.dayofyear
    return combined_df

# =============================================================================
# 更新所有網格點的當年氣象資料（結合預報） - 泰國版
# 每次更新前會檢查該網格點是否在72小時內已更新過，
# 且整個更新作業不超過1小時。
# 儲存資料至 weather_data_tmp_thailand
# =============================================================================
def update_weather_data_thailand(forecast_days=16, batch_size=10):
    grid_path = "./1_grid_points/taiwan_grid.csv"
    output_dir = "./weather_data_tmp_thailand"
    os.makedirs(output_dir, exist_ok=True)
    
    grid_df = pd.read_csv(grid_path)
    logging.info(f"讀取網格點資料，共 {len(grid_df)} 筆。")
    points_to_update = []
    meta_files = {}
    start_time = time.time()  # 記錄開始更新的時間
    for index, row in grid_df.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        file_name = f"lat_{lat}_lon_{lon}.csv"
        meta_file = os.path.join(output_dir, f"lat_{lat}_lon_{lon}.meta")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                last_update = float(f.read().strip())
            # 若該點在72小時內已更新，則跳過
            if time.time() - last_update < 72 * 3600:
                logging.info(f"{file_name} 在 72 小時內已更新，跳過。")
                continue
        points_to_update.append((lat, lon))
        meta_files[(lat, lon)] = meta_file
        # 若總執行時間超過1小時，則停止後續更新
        if time.time() - start_time > 3600:
            logging.info("已達到更新一小時的限制，停止更新。")
            break
    if not points_to_update:
        logging.info("所有網格點皆在 72 小時內更新，無需下載。")
        return
    
    lats, lons = zip(*points_to_update)
    current_year = datetime.now().year
    today = date.today()
    exec_date = pd.Timestamp.now().normalize()
    if current_year == datetime.now().year:
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = f"{current_year}-12-31"
    start_date = f"{current_year}-01-01"
    
    # 批次下載歷史資料
    hist_responses = batch_fetch_historical(lats, lons, start_date, end_date, batch_size=batch_size)
    hist_data_dict = {}
    for i, response in enumerate(hist_responses):
        try:
            df = process_historical_response(response)
            hist_data_dict[(lats[i], lons[i])] = df
            logging.info(f"歷史資料處理成功：({lats[i]}, {lons[i]})，筆數：{len(df)}")
        except Exception as e:
            logging.error(f"處理歷史資料失敗：({lats[i]}, {lons[i]})，錯誤：{e}")
    
    # 批次下載預報資料
    fc_responses = batch_fetch_forecast(lats, lons, forecast_days, batch_size=batch_size)
    fc_data_dict = {}
    for i, response in enumerate(fc_responses):
        try:
            df = process_forecast_response(response)
            fc_data_dict[(lats[i], lons[i])] = df
            logging.info(f"預報資料處理成功：({lats[i]}, {lons[i]})，筆數：{len(df)}")
        except Exception as e:
            logging.error(f"處理預報資料失敗：({lats[i]}, {lons[i]})，錯誤：{e}")
    
    # 合併歷史與預報資料，並儲存 CSV
    for point in points_to_update:
        lat, lon = point
        if point not in hist_data_dict:
            logging.error(f"缺少歷史資料：({lat}, {lon})")
            continue
        hist_df = hist_data_dict[point]
        fc_df = fc_data_dict.get(point, pd.DataFrame())
        combined_df = merge_weather_data(hist_df, fc_df, current_year, forecast_days)
        file_name = f"lat_{lat}_lon_{lon}.csv"
        file_path = os.path.join(output_dir, file_name)
        combined_df.to_csv(file_path, index=False)
        meta_path = meta_files[point]
        with open(meta_path, "w") as f:
            f.write(str(time.time()))
        logging.info(f"儲存 {file_name} 至 {file_path}")
        time.sleep(5)

# =============================================================================
# 預報功能：從指定資料夾載入氣象資料並依日期預測 - 泰國版
# 輸入資料夾預設為 weather_data_tmp_thailand，
# 輸出結果儲存至 2_predictions_thailand
# =============================================================================
def run_forecast_from_weather_data_thailand(input_folder="./weather_data_tmp_thailand"):
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
        df["latitude"] = lat_val
        df["longitude"] = lon_val
        for idx, row in df.iterrows():
            feature_dict = {k: row[k] for k in expected_daily_keys if k in row}
            for k in expected_cum_keys:
                if k in row:
                    feature_dict[k] = round(row[k], 4)
            X = pd.DataFrame([feature_dict])
            individual_preds = {}
            cf_preds = []
            cg_preds = []
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
                try:
                    proba = model.predict_proba(X)[:, 1][0]
                    individual_preds[model_name] = round(proba, 3)
                    if model_name.startswith("cf_"):
                        cf_preds.append(proba)
                    elif model_name.startswith("cg_"):
                        cg_preds.append(proba)
                except Exception as e:
                    logging.error(f"模型 {model_name} 在 {row['date']} 預測失敗：{e}")
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
    
    output_dir = "./2_predictions_thailand"
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
# 主入口：使用 argparse 決定執行模式
# 新增模式: update_thailand (更新泰國資料), forecast_thailand (泰國資料預報)
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Weather Data and Forecast Runner")
    parser.add_argument("mode", choices=["update", "forecast", "update_historical", "forecast_historical", "update_thailand", "forecast_thailand"],
                        help="模式: update / forecast / update_historical / forecast_historical / update_thailand / forecast_thailand")
    parser.add_argument("--year", type=int, help="指定歷史資料的年份（update_historical 或 forecast_historical 必須）")
    parser.add_argument("--forecast_days", type=int, default=16, help="預報天數（預設 16 天）")
    parser.add_argument("--batch_size", type=int, default=10, help="批次下載點數（預設 10 個）")
    args = parser.parse_args()
    
    if args.mode == "update":
        logging.info("開始更新當年氣象資料（結合預報）...")
        update_weather_data(forecast_days=args.forecast_days, batch_size=args.batch_size)
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
        update_historical_weather_data(args.year, batch_size=args.batch_size)
        logging.info(f"{args.year} 年歷史氣象資料更新完成。")
    elif args.mode == "forecast_historical":
        if args.year is None:
            logging.error("請提供目標年份 --year (例如 --year 2024)")
            return
        logging.info(f"開始依 {args.year} 年歷史氣象資料進行預報...")
        run_forecast_from_weather_data(input_folder=f"./weather_data_historical/{args.year}")
        logging.info(f"{args.year} 年歷史預報結果已儲存至 ./2_predictions/。")
    elif args.mode == "update_thailand":
        logging.info("開始更新泰國氣象資料（結合預報）...")
        update_weather_data_thailand(forecast_days=args.forecast_days, batch_size=args.batch_size)
        logging.info("泰國氣象資料更新完成。")
    elif args.mode == "forecast_thailand":
        logging.info("開始依泰國氣象資料進行預報...")
        run_forecast_from_weather_data_thailand(input_folder="./weather_data_tmp_thailand")
        logging.info("泰國預報結果已儲存至 ./2_predictions_thailand/。")
    else:
        logging.error("未知的模式")
        return

if __name__ == "__main__":
    main()
