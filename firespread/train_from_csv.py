# train_from_csv.py
# : FireBench CSV 기반 모델 학습 및 평가

import os
import pandas as pd
import numpy as np
from model_trainer import train_model, evaluate_model
from sklearn.model_selection import train_test_split
from firebench_loader import load_all_samples_from_firebench, save_dataset_as_csv

# CSV 파일에서 X, y data 추출
def load_data_from_csv(filepath: str):
    df = pd.read_csv(filepath)
    expected_columns = ["wind_dir_local", "wind_speed", "slope_deg", "spread_dir_local", "spread_speed"]
    if list(df.columns) != expected_columns:
        raise ValueError(
            "CSV 스키마가 현재 FireBench 로더와 다릅니다. "
            "기존 CSV를 삭제한 뒤 다시 생성하세요. "
            f"필요한 열: {expected_columns}"
        )
    X = df[["wind_dir_local", "wind_speed", "slope_deg"]].values
    y = df[["spread_dir_local", "spread_speed"]].values
    return X, y

def main():
    csv_path = "firebench_dataset.csv"
    model_path = "fire_model.pkl"

    if not os.path.exists(csv_path):
        print("⚙️ CSV 파일 없음. 생성 중...")
        X, y = load_all_samples_from_firebench()
        save_dataset_as_csv(X, y, csv_path)
    else:
        print("📂 기존 CSV 로딩 중...")
        X, y = load_data_from_csv(csv_path)

    print("📊 데이터셋 크기:", X.shape)
    print("📦 학습/검증 세트 분리...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🧠 모델 학습 중...")
    model = train_model(X_train, y_train, save_path=model_path)

    print("📈 평가 중...")
    metrics = evaluate_model(model, X_test, y_test)
    for k, v in metrics.items():
        print(f"{k}: spread_dir_local={v[0]:.4f}, spread_speed={v[1]:.4f}")

    print(f"✅ 학습 완료! 모델 저장 경로: {model_path}")

if __name__ == "__main__":
    main()
