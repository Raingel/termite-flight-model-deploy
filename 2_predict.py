# %%
import os
import glob
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
import openmeteo_requests
from openmeteo_requests.Client import OpenMeteoRequestsError
import json
import time
import logging
import argparse

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =============================================================================
# 批次下載歷史氣象資料：利用 ERA5_land（ECMWF Seamless）
# （已改為在 update_weather_data 中分批處理，因此不再使用此函式）
# =============================================================================

# =============================================================================
# 批次下載預報氣象資料：利用 ECMWF IFS 預報資料
# （已改為在 update_weather_data 中分批處理，因此不再使用此函式）
# =============================================================================

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

def merge_weather_data(hist_df, fc_df, current_year, forecast_days):
    exec_date = pd.Timestamp.now().normalize()
    start_date = pd.Timestamp(f"{current_year}-01-01").tz_localize('UTC')
    end_date = (exec_date + pd.Timedelta(days=forecast_days)).tz_localize('UTC')

    hist_df = hist_df.dropna()
    combined_df = pd.concat([hist_df, fc_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset="date", keep="first")
    combined_df = combined_df[(combined_df["date"] >= start_date) & (combined_df["date"] <= end_date)]
    combined_df = combined_df.sort_values("date").reset_index(drop=True)
    combined_df["elevation"] = combined_df["elevation"].fillna(combined_df["elevation"].iloc[0])

    cum_vars = ["temperature_2m_mean", "apparent_temperature_mean", "daylight_duration",
                "sunshine_duration", "precipitation_sum", "rain_sum", "precipitation_hours",
                "shortwave_radiation_sum", "et0_fao_evapotranspiration"]
    for var in cum_vars:
        combined_df[f"cumulative_{var}"] = combined_df[var].cumsum()
    combined_df["day"] = combined_df["date"].dt.dayofyear
    return combined_df

# =============================================================================
# 更新所有網格點的當年氣象資料（結合預報），以批次方式下載
# 存成 CSV 至 ./weather_data_tmp/
# =============================================================================
def update_weather_data(forecast_days=16, batch_size=10):
    HIST_URL = "https://archive-api.open-meteo.com/v1/archive"
    FC_URL = "https://api.open-meteo.com/v1/forecast"
    TZ = "Asia/Singapore"

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
        if os.path.exists(meta_file):
            last_update = float(open(meta_file).read().strip())
            if time.time() - last_update < 20*3600:
                logging.info(f"lat_{lat}_lon_{lon}.csv 在 20 小時內已更新，跳過。")
                continue
        points_to_update.append((lat, lon))
        meta_files[(lat, lon)] = meta_file

    if not points_to_update:
        logging.info("所有網格點皆在 20 小時內更新，無需下載。")
        return

    client = openmeteo_requests.Client()
    current_year = datetime.now().year
    today = date.today()
    if current_year == today.year:
        end_date_str = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        end_date_str = f"{current_year}-12-31"
    start_date_str = f"{current_year}-01-01"

    # 按 batch 處理
    for i in range(0, len(points_to_update), batch_size):
        chunk = points_to_update[i:i+batch_size]
        lats, lons = zip(*chunk)

        # 歷史資料參數
        hist_params = {
            "latitude": list(lats), "longitude": list(lons),
            "start_date": start_date_str, "end_date": end_date_str,
            "daily": [
                "temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                "apparent_temperature_max","apparent_temperature_min","apparent_temperature_mean",
                "sunrise","sunset","daylight_duration","sunshine_duration",
                "precipitation_sum","rain_sum","precipitation_hours",
                "wind_speed_10m_max","wind_gusts_10m_max","wind_direction_10m_dominant",
                "shortwave_radiation_sum","et0_fao_evapotranspiration"
            ],
            "timezone": TZ, "models": "era5_seamless"
        }

        logging.info(f"批次下載歷史資料：第 {i+1} 到 {i+len(chunk)} 個點")
        try:
            hist_resps = client.weather_api(HIST_URL, params=hist_params)
        except OpenMeteoRequestsError as e:
            logging.error(f"歷史資料下載失敗，API 限制或錯誤：{e}")
            break

        # 處理歷史資料
        hist_dfs = []
        for idx, resp in enumerate(hist_resps):
            try:
                df_hist = process_historical_response(resp)
                hist_dfs.append(df_hist)
            except Exception as e:
                logging.error(f"處理歷史回應失敗 (點 {chunk[idx]})：{e}")
                hist_dfs.append(pd.DataFrame())  # 空 DF 以保形

        time.sleep(60)

        # 預報資料參數
        fc_params = {
            "latitude": list(lats), "longitude": list(lons),
            "daily": [
                "temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                "apparent_temperature_max","apparent_temperature_min","apparent_temperature_mean",
                "daylight_duration","sunshine_duration","precipitation_sum","rain_sum",
                "precipitation_hours","wind_speed_10m_max","wind_gusts_10m_max",
                "wind_direction_10m_dominant","shortwave_radiation_sum","et0_fao_evapotranspiration"
            ],
            "timezone": TZ, "past_days": 10, "forecast_days": forecast_days, "models": "best_match"
        }

        logging.info(f"批次下載預報資料：第 {i+1} 到 {i+len(chunk)} 個點")
        try:
            fc_resps = client.weather_api(FC_URL, params=fc_params)
        except OpenMeteoRequestsError as e:
            logging.error(f"預報資料下載失敗，API 限制或錯誤：{e}")
            break

        # 處理預報資料
        fc_dfs = []
        for idx, resp in enumerate(fc_resps):
            try:
                df_fc = process_forecast_response(resp)
                fc_dfs.append(df_fc)
            except Exception as e:
                logging.error(f"處理預報回應失敗 (點 {chunk[idx]})：{e}")
                fc_dfs.append(pd.DataFrame())

        time.sleep(60)

        # 合併並寫出每一個點的 CSV
        for idx, (lat, lon) in enumerate(chunk):
            if hist_dfs[idx].empty or fc_dfs[idx].empty:
                logging.warning(f"跳過合併，缺少歷史或預報資料：({lat}, {lon})")
                continue
            combined = merge_weather_data(hist_dfs[idx], fc_dfs[idx], current_year, forecast_days)
            fname = f"lat_{lat}_lon_{lon}.csv"
            fpath = os.path.join(output_dir, fname)
            combined.to_csv(fpath, index=False)

            # 更新 meta 檔時間戳
            with open(meta_files[(lat, lon)], "w") as mf:
                mf.write(str(time.time()))
            logging.info(f"已儲存：{fpath}")

            time.sleep(5)

# =============================================================================
# 更新指定歷史年度的氣象資料（僅歷史），以批次方式下載
# 存成 CSV 至 ./weather_data_historical/{target_year}/
# =============================================================================
def update_historical_weather_data(target_year, batch_size=10):
    grid_path = "./1_grid_points/taiwan_grid.csv"
    output_dir = f"./weather_data_historical/{target_year}"
    os.makedirs(output_dir, exist_ok=True)

    grid_df = pd.read_csv(grid_path)
    logging.info(f"[{target_year}] 讀取網格點資料，共 {len(grid_df)} 筆。")
    points_to_update = []
    meta_files = {}
    for _, row in grid_df.iterrows():
        lat, lon = row["lat"], row["lon"]
        meta_file = os.path.join(output_dir, f"lat_{lat}_lon_{lon}.meta")
        if os.path.exists(meta_file):
            last_update = float(open(meta_file).read().strip())
            if time.time() - last_update < 24*3600:
                logging.info(f"[{target_year}] lat_{lat}_lon_{lon}.csv 在 24 小時內已更新，跳過。")
                continue
        points_to_update.append((lat, lon))
        meta_files[(lat, lon)] = meta_file

    if not points_to_update:
        logging.info(f"[{target_year}] 所有網格點皆在 24 小時內更新，無需下載。")
        return

    HIST_URL = "https://archive-api.open-meteo.com/v1/archive"
    TZ = "Asia/Singapore"
    client = openmeteo_requests.Client()
    start_date = f"{target_year}-01-01"
    end_date = f"{target_year}-12-31"

    for i in range(0, len(points_to_update), batch_size):
        chunk = points_to_update[i:i+batch_size]
        lats, lons = zip(*chunk)
        params = {
            "latitude": list(lats), "longitude": list(lons),
            "start_date": start_date, "end_date": end_date,
            "daily": [
                "temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                "apparent_temperature_max","apparent_temperature_min","apparent_temperature_mean",
                "sunrise","sunset","daylight_duration","sunshine_duration",
                "precipitation_sum","rain_sum","precipitation_hours",
                "wind_speed_10m_max","wind_gusts_10m_max","wind_direction_10m_dominant",
                "shortwave_radiation_sum","et0_fao_evapotranspiration"
            ],
            "timezone": TZ, "models": "era5_seamless"
        }
        logging.info(f"[{target_year}] 批次下載歷史資料：第 {i+1} 到 {i+len(chunk)} 個點")
        try:
            resps = client.weather_api(HIST_URL, params=params)
        except OpenMeteoRequestsError as e:
            logging.error(f"[{target_year}] 歷史資料下載失敗：{e}")
            break

        for idx, resp in enumerate(resps):
            lat, lon = chunk[idx]
            try:
                df = process_historical_response(resp)
                cum_vars = ["temperature_2m_mean","apparent_temperature_mean","daylight_duration",
                            "sunshine_duration","precipitation_sum","rain_sum","precipitation_hours",
                            "shortwave_radiation_sum","et0_fao_evapotranspiration"]
                for var in cum_vars:
                    df[f"cumulative_{var}"] = df[var].cumsum()
                df["day"] = pd.to_datetime(df["date"]).dt.dayofyear
                df["latitude"], df["longitude"] = lat, lon
                fpath = os.path.join(output_dir, f"lat_{lat}_lon_{lon}.csv")
                df.to_csv(fpath, index=False)
                with open(meta_files[(lat, lon)], "w") as mf:
                    mf.write(str(time.time()))
                logging.info(f"[{target_year}] 儲存：{fpath}")
            except Exception as e:
                logging.error(f"[{target_year}] 處理/儲存失敗 ({lat},{lon})：{e}")
        time.sleep(5)

# =============================================================================
# 預報功能：從指定資料夾載入氣象資料並依日期預測
# =============================================================================
def run_forecast_from_weather_data(input_folder="./weather_data_tmp"):
    logging.info("開始讀取預先訓練的模型...")
    model_files = glob.glob(os.path.join("models", "*_model.pkl"))
    models = {}
    for mf in model_files:
        name = os.path.splitext(os.path.basename(mf))[0]
        try:
            with open(mf, "rb") as f:
                models[name] = pickle.load(f)
            logging.info(f"成功讀取模型：{name}")
        except Exception as e:
            logging.error(f"讀取模型 {mf} 錯誤: {e}")

    expected_daily = ["elevation","temperature_2m_max","temperature_2m_min","temperature_2m_mean",
                      "apparent_temperature_max","apparent_temperature_min","apparent_temperature_mean",
                      "daylight_duration","precipitation_sum","rain_sum","precipitation_hours",
                      "wind_speed_10m_max","wind_gusts_10m_max","shortwave_radiation_sum",
                      "et0_fao_evapotranspiration","latitude","longitude","day"]
    cum_vars = ["temperature_2m_mean","apparent_temperature_mean","daylight_duration",
                "sunshine_duration","precipitation_sum","rain_sum","precipitation_hours",
                "shortwave_radiation_sum","et0_fao_evapotranspiration"]
    expected_cum = [f"cumulative_{v}" for v in cum_vars]

    weather_files = glob.glob(os.path.join(input_folder, "*.csv"))
    logging.info(f"找到 {len(weather_files)} 個氣象檔案於 {input_folder}。")
    all_results = []
    for wf in weather_files:
        try:
            df = pd.read_csv(wf)
            logging.info(f"讀取 {wf}，共 {len(df)} 筆。")
        except Exception as e:
            logging.error(f"讀取 {wf} 失敗：{e}")
            continue

        base = os.path.basename(wf)
        try:
            _, lat_str, _, lon_str = base.replace(".csv","").split("_")
            lat_val, lon_val = float(lat_str), float(lon_str)
        except Exception as e:
            logging.error(f"解析檔名 {base} 失敗：{e}")
            continue

        df["latitude"], df["longitude"] = lat_val, lon_val
        for _, row in df.iterrows():
            feat = {k: row[k] for k in expected_daily if k in row}
            for k in expected_cum:
                if k in row:
                    feat[k] = round(row[k], 4)
            X = pd.DataFrame([feat])

            individual_preds, cf_preds, cg_preds = {}, [], []
            weights = {
                "cf_glm_model":0.10,"cf_lda_model":0.15,"cf_nn_model":0.10,
                "cf_rf_model":0.10,"cf_svm_model":0.55,"cg_glm_model":0.10,
                "cg_lda_model":0.10,"cg_nn_model":0.10,"cg_rf_model":0.55,
                "cg_svm_model":0.15
            }
            for name, model in models.items():
                if name not in weights:
                    logging.error(f"模型 {name} 未指定權重，跳過。")
                    continue
                try:
                    p = model.predict_proba(X)[:,1][0]
                    individual_preds[name] = round(p,3)
                    if name.startswith("cf_"): cf_preds.append(p)
                    if name.startswith("cg_"): cg_preds.append(p)
                except Exception as e:
                    logging.error(f"{name} 在 {row['date']} 預測失敗：{e}")

            ensemble_cf = float(np.mean(cf_preds)) if cf_preds else None
            ensemble_cg = float(np.mean(cg_preds)) if cg_preds else None
            interaction = ensemble_cf * ensemble_cg if (ensemble_cf and ensemble_cf>=0.5 and ensemble_cg and ensemble_cg>=0.5) else 0.0

            all_results.append({
                "date": row["date"],
                "individual_predictions": individual_preds,
                "ensemble": {"cf": ensemble_cf, "cg": ensemble_cg, "interaction_score": interaction},
                "features": {"latitude": lat_val, "longitude": lon_val}
            })

    logging.info(f"共產生 {len(all_results)} 筆預測結果。")
    forecasts = {}
    for item in all_results:
        ds = pd.to_datetime(item["date"]).strftime("%Y-%m-%d")
        forecasts.setdefault(ds, []).append(item)

    out_dir = "./2_predictions"
    os.makedirs(out_dir, exist_ok=True)
    for ds, items in forecasts.items():
        path = os.path.join(out_dir, f"{ds}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        logging.info(f"儲存預報結果：{path}")

# %%
def main():
    parser = argparse.ArgumentParser(description="Weather Data and Forecast Runner")
    parser.add_argument("mode", choices=["update","forecast","update_historical","forecast_historical"],
                        help="模式: update / forecast / update_historical / forecast_historical")
    parser.add_argument("--year", type=int, help="指定歷史資料年份，update_historical 或 forecast_historical 必須提供")
    parser.add_argument("--forecast_days", type=int, default=16, help="預報天數（預設 16）")
    parser.add_argument("--batch_size", type=int, default=10, help="批次下載點數（預設 10）")
    args = parser.parse_args()

    if args.mode == "update":
        logging.info("開始更新當年氣象資料（結合預報）...")
        update_weather_data(forecast_days=args.forecast_days, batch_size=args.batch_size)
        logging.info("當年氣象資料更新完成。")
    elif args.mode == "forecast":
        logging.info("開始依當年氣象資料預報...")
        run_forecast_from_weather_data(input_folder="./weather_data_tmp")
        logging.info("預報完成並儲存於 ./2_predictions/")
    elif args.mode == "update_historical":
        if not args.year:
            logging.error("請提供 --year 參數 (例如 --year 2024)")
            return
        logging.info(f"開始更新 {args.year} 年歷史氣象資料...")
        update_historical_weather_data(args.year, batch_size=args.batch_size)
        logging.info(f"{args.year} 年歷史資料更新完成。")
    elif args.mode == "forecast_historical":
        if not args.year:
            logging.error("請提供 --year 參數 (例如 --year 2024)")
            return
        folder = f"./weather_data_historical/{args.year}"
        logging.info(f"開始依 {args.year} 年歷史資料預報...")
        run_forecast_from_weather_data(input_folder=folder)
        logging.info(f"{args.year} 年歷史預報完成並儲存於 ./2_predictions/")
    else:
        logging.error("未知模式")

if __name__ == "__main__":
    main()
