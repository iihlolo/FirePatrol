# model_trainer.py
# : 산불 확산 예측 모델 학습 및 성능 평가

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# 풍향과 경사 방향 간 벡터 alignment (cosine of angle difference)
# FireBench 입력: 로컬 풍향(도), 풍속(m/s), 경사각(도)
def add_engineered_features(X: np.ndarray) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != 3:
        raise ValueError("X는 [wind_dir_local, wind_speed, slope_deg] 3개 열이어야 합니다.")

    wind_dir_rad = np.deg2rad(X[:, 0])
    return np.column_stack(
        [np.cos(wind_dir_rad), np.sin(wind_dir_rad), X[:, 1], X[:, 2]]
    )

# 산불 확산 방향과 속도를 예측하는 회귀 모델 학습
def train_model(X: np.ndarray, y: np.ndarray, save_path: str = "fire_model.pkl"):

    X = add_engineered_features(X)

    base_model = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X, y)

    joblib.dump(model, save_path)

    return model

# testset에 대해 예측 정확도 평가
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):

    X_test = add_engineered_features(X_test)
    y_pred = model.predict(X_test)

    # 평가 지표
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    return {
        "MAE (deg, m/s)": mae,
        "RMSE (deg, m/s)": rmse
    }