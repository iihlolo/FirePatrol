# fire_predictor.py
# : 위경도·경사 방향·기상·지형 데이터를 결합해 산불 확산 방향과 속도를 예측

import numpy as np
import joblib
import os
from pathlib import Path
from dotenv import load_dotenv
from model_trainer import add_engineered_features
from terrain_reader import get_slope_from_geotiff
from weather_api import get_wind_data

def predict_fire_at(
    lat: float,
    lon: float,
    slope_axis_azimuth_deg: float,
    geotiff_path: str,
    weather_api_key: str,
    model_path: str = "fire_model.pkl"
) -> dict:
    """Predict spread using the geographic bearing of the FireBench +x axis.

    The caller must provide that axis bearing in degrees clockwise from north.
    The local +y axis is assumed to be 90 degrees clockwise from local +x.
    FireBench itself does not provide this geographic orientation.
    """
    slope_deg = get_slope_from_geotiff(geotiff_path, lat, lon)
    wind_from_dir, wind_speed = get_wind_data(lat, lon, weather_api_key)
    model = load_model(model_path)

    # 기상 API는 바람이 불어오는 방위각을 반환하므로 진행 방향으로 180° 회전합니다.
    wind_to_azimuth = (wind_from_dir + 180.0) % 360.0
    wind_dir_local = (wind_to_azimuth - slope_axis_azimuth_deg) % 360.0
    result = predict_fire_spread(model, wind_dir_local, wind_speed, slope_deg)

    # 로컬 +x축의 지리 방위각을 기준으로 로컬 확산 방향을 지리 방위각으로 변환합니다.
    spread_dir_azimuth = (slope_axis_azimuth_deg + result["spread_dir"]) % 360.0

    return {
        "gps": (lat, lon),
        "wind_dir": wind_from_dir,
        "wind_speed": wind_speed,
        "slope_deg": slope_deg,
        "slope_axis_azimuth_deg": slope_axis_azimuth_deg,
        "spread_dir_local": result["spread_dir"],
        "spread_dir": spread_dir_azimuth,
        "spread_speed": result["spread_speed"]
    }

# 학습된 모델 loading
def load_model(model_path: str = "fire_model.pkl"):

    return joblib.load(model_path)

# 입력된 조건에 대해 산불 확산 방향 및 속도를 예측
def predict_fire_spread(model, wind_dir_local_deg: float, wind_speed: float, slope_deg: float) -> dict:
    X = np.array([[wind_dir_local_deg, wind_speed, slope_deg]], dtype=np.float32)
    pred = model.predict(add_engineered_features(X))[0]

    return {
        "spread_dir": float(pred[0]),
        "spread_speed": float(pred[1])
    }

def main():
    # 예시용 설정값
    lat = 37.4563
    lon = 126.7052
    slope_axis_azimuth_deg = 135.0 # 지리 방위각 대입
    geotiff_path = "Slope_All.tif"
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    weather_api_key = os.getenv("WEATHER_API_KEY")
    if not weather_api_key:
        raise RuntimeError("프로젝트 루트 .env에 WEATHER_API_KEY를 설정하세요.")
    model_path = "fire_model.pkl"

    print("🔥 산불 확산 예측 테스트 시작 🔥")
    result = predict_fire_at(
        lat=lat,
        lon=lon,
        slope_axis_azimuth_deg=slope_axis_azimuth_deg,
        geotiff_path=geotiff_path,
        weather_api_key=weather_api_key,
        model_path=model_path
    )

    print("\n입력 위치 (GPS):", result["gps"])
    print(f"풍향: {result['wind_dir']:.1f}°, 풍속: {result['wind_speed']:.2f} m/s")
    print(f"경사도: {result['slope_deg']:.1f}°, +x축 방위: {result['slope_axis_azimuth_deg']:.1f}°")
    print(f"\n예상 확산 방향: {result['spread_dir']:.1f}°")
    print(f"예상 확산 속도: {result['spread_speed']:.2f} m/s")

if __name__ == "__main__":
    main()
