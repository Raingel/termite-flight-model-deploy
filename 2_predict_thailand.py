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
import sys

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# 轉換 API 回應成歷史資料的 DataFrame
# =============================================================================
def process_historical_response(response):
    daily = response.Daily()
    daily_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }
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

# =============================================================================
# 轉換 API 回應成預報資料的 DataFrame
# =============================================================================
def process_forecast_response(response):
    daily = response.Daily()
    forecast_data = {
        "date": pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
    }
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
# 合併歷史與預報資料，並計算累積特徵
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
# 更新泰國網格點的當年氣象資料（歷史＋預報）
# - 先讀取所有網格點資料（檔案：grid_points_thailand.csv）
# - 檢查每個點是否在update_freq小時內已更新，若是則跳過
# - 將需要更新的網格點依 batch_size 分批處理（例如每批50個）
# - 對同一批次內的網格點先下載歷史資料，再下載預報資料（若任一 API 呼叫失敗則中斷執行）
# - 合併該網格點的歷史與預報資料後，存成 CSV 並更新 meta 檔案
# =============================================================================
def update_weather_data_thailand(forecast_days=16, batch_size=50):
    grid_path = "./1_grid_points/grid_points_thailand.csv"
    output_dir = "./weather_data_tmp_thailand"
    os.makedirs(output_dir, exist_ok=True)
    update_freq = 24
    grid_df = pd.read_csv(grid_path)
    logging.info(f"讀取網格點資料，共 {len(grid_df)} 筆。")
    points_to_update = []
    meta_files = {}
    start_time = time.time()  # 記錄整體更新開始的時間
    for index, row in grid_df.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        file_name = f"lat_{lat}_lon_{lon}.csv"
        meta_file = os.path.join(output_dir, f"lat_{lat}_lon_{lon}.meta")
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                last_update = float(f.read().strip())
            # 若該點在?小時內已更新，則跳過
            if time.time() - last_update < update_freq * 3600:
                logging.info(f"{file_name} 在 {update_freq} 小時內已更新，跳過。")
                continue
        points_to_update.append((lat, lon))
        meta_files[(lat, lon)] = meta_file
        # 如果整體更新時間超過1小時則停止後續更新
        if time.time() - start_time > 3600:
            logging.info("已達到更新一小時的限制，停止更新。")
            break
    if not points_to_update:
        logging.info("所有網格點皆在 {update_freq} 小時內更新，無需下載。")
        return

    # 設定日期參數（以當年資料為例）
    current_year = datetime.now().year
    today = date.today()
    if current_year == datetime.now().year:
        end_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date = f"{current_year}-12-31"
    start_date = f"{current_year}-01-01"
    
    client = openmeteo_requests.Client()
    
    # 分批處理下載，每批處理 batch_size 個網格點
    for batch_start in range(0, len(points_to_update), batch_size):
        batch_points = points_to_update[batch_start : batch_start + batch_size]
        batch_lats = [pt[0] for pt in batch_points]
        batch_lons = [pt[1] for pt in batch_points]
        
        # ---------------------------------------------------------------------
        # 下載該批次的歷史資料
        # ---------------------------------------------------------------------
        hist_params = {
            "latitude": batch_lats,
            "longitude": batch_lons,
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
        logging.info(f"批次下載歷史資料：處理第 {batch_start+1} 到 {batch_start+len(batch_points)} 個點")
        try:
            hist_responses = client.weather_api("https://archive-api.open-meteo.com/v1/archive", params=hist_params)
        except Exception as e:
            logging.error(f"API 下載錯誤：{e}。立即中斷執行。")
            sys.exit(0)
        logging.info("等待60秒以避免 API 限制")
        time.sleep(60)
        
        # ---------------------------------------------------------------------
        # 下載該批次的預報資料
        # ---------------------------------------------------------------------
        fc_params = {
            "latitude": batch_lats,
            "longitude": batch_lons,
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
        logging.info(f"批次下載預報資料：處理第 {batch_start+1} 到 {batch_start+len(batch_points)} 個點")
        try:
            fc_responses = client.weather_api("https://api.open-meteo.com/v1/forecast", params=fc_params)
        except Exception as e:
            logging.error(f"API 下載錯誤：{e}。立即中斷執行。")
            sys.exit(0)
        logging.info("等待60秒以避免 API 限制")
        time.sleep(60)
        
        # ---------------------------------------------------------------------
        # 針對該批次內的每一個網格點，合併資料並存檔
        # ---------------------------------------------------------------------
        for i, point in enumerate(batch_points):
            lat, lon = point
            try:
                hist_df = process_historical_response(hist_responses[i])
            except Exception as e:
                logging.error(f"處理歷史資料失敗：({lat}, {lon})，錯誤：{e}")
                continue
            try:
                fc_df = process_forecast_response(fc_responses[i])
            except Exception as e:
                logging.error(f"處理預報資料失敗：({lat}, {lon})，錯誤：{e}")
                fc_df = pd.DataFrame()
            try:
                combined_df = merge_weather_data(hist_df, fc_df, current_year, forecast_days)
            except Exception as e:
                logging.error(f"合併資料失敗：({lat}, {lon})，錯誤：{e}")
                continue
            file_name = f"lat_{lat}_lon_{lon}.csv"
            file_path = os.path.join(output_dir, file_name)
            combined_df.to_csv(file_path, index=False)
            meta_path = meta_files[(lat, lon)]
            with open(meta_path, "w") as f:
                f.write(str(time.time()))
            logging.info(f"儲存 {file_name} 至 {file_path}")
            time.sleep(5)  # 每個網格點儲存後稍作等待

    logging.info("所有批次更新完成。")

# =============================================================================
# 預報功能：讀取泰國氣象資料並依日期預測
# （此部分保持原有邏輯）
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
    today = pd.Timestamp.now(tz='Asia/Taipei').normalize()
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
        # 將日期轉換成 datetime 型別
        df["date"] = pd.to_datetime(df["date"])
        # 僅保留「未來資料」與過去 21 天的資料（作為預測的上下文）
        df = df[df["date"] >= (today - pd.Timedelta(days=21))]
        # 若只希望預測未來資料，則進一步過濾出日期 >= 今日的部分
        df_forecast = df[df["date"] >= today]
        
        for idx, row in df_forecast.iterrows():
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
                "date": row["date"].strftime("%Y-%m-%d"),
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
        ds = res["date"]
        if ds not in forecasts_by_date:
            forecasts_by_date[ds] = []
        forecasts_by_date[ds].append(res)
    
    output_dir = "./2_predictions_thailand"
    os.makedirs(output_dir, exist_ok=True)
    for ds, results_list in forecasts_by_date.items():
        out_path = os.path.join(output_dir, f"{ds}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_list, f, ensure_ascii=False, indent=2)
        logging.info(f"儲存預報結果 {ds} 至 {out_path}")

# =============================================================================
# 主入口：僅提供 update_thailand 與 forecast_thailand 模式
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Weather Data and Forecast Runner (Thailand)")
    parser.add_argument("mode", choices=["update_thailand", "forecast_thailand"],
                        help="模式: update_thailand / forecast_thailand")
    parser.add_argument("--forecast_days", type=int, default=16, help="預報天數（預設 16 天）")
    parser.add_argument("--batch_size", type=int, default=50, help="批次下載網格點數（預設 50 個）")
    args = parser.parse_args()
    
    if args.mode == "update_thailand":
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
