# FirePatrol: Fire Spread Pipeline

FirePatrol은 CCTV 영상에서 산불 발생을 감지하고, 산불의 확산 방향과 속도를 예측해 대시보드에 제공함으로써 초기 대응을 돕는 것을 목표로 합니다.

이 저장소는 전체 FirePatrol 중 **산불 확산 예측 파이프라인**을 다룹니다. FireBench 시뮬레이션 자료를 이용해 회귀 모델을 학습하고, 위경도·기상·지형 정보를 받아 확산 방향과 속도를 추정합니다. CCTV 영상 분석, 화재 감지, API 서버와 대시보드 구현은 이 저장소의 현재 범위에 포함되지 않습니다.

## 기능

- 공개 FireBench Zarr 데이터를 Google Cloud Storage에서 읽고 학습 샘플을 추출합니다.
- 풍향, 풍속, 경사각을 입력으로 사용하고 로컬 확산 방향과 속도를 예측하도록 Random Forest 모델을 학습합니다.
- CSV 데이터셋과 학습 모델을 저장하고, 홀드아웃 데이터로 MAE와 RMSE를 출력합니다.
- 추론에서는 입력 위경도에 해당하는 GeoTIFF 경사도와 기상청 초단기실황 풍향·풍속을 조회해 저장된 모델에 전달합니다.

## 처리 흐름

```text
FireBench 공개 Zarr
	│
	▼
firebench_loader.py ──▶ firebench_dataset.csv
	│                         │
	└─────────────────────────▼
			 train_from_csv.py
				│
				▼
			 fire_model.pkl

위경도 + GeoTIFF + 기상청 API ──▶ fire_predictor.py
					  │
					  ▼
			      확산 방향·속도 출력
```

## 디렉터리 구조

```text
.
├── firespread/
│   ├── firebench_loader.py   # FireBench 경로 탐색, Zarr 샘플 로드와 특징 추출
│   ├── model_trainer.py      # 특징 변환, Random Forest 학습과 평가
│   ├── train_from_csv.py     # CSV 로드/생성, 학습과 검증 실행
│   ├── fire_predictor.py     # 위경도·기상·지형 입력을 이용한 추론
│   ├── weather_api.py        # 기상청 초단기실황 조회와 격자 변환
│   └── terrain_reader.py    # GeoTIFF에서 지점 경사도 조회
├── notebooks/
│   └── firebench_analysis.ipynb  # FireBench 구조와 한국 적용성 분석 노트북
├── requirements.txt
└── README.md
```

## 설치

Python 3.11을 기준으로 작성했습니다. 프로젝트 루트에서 가상환경을 만들고 의존성을 설치합니다.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell에서는 가상환경 활성화 명령을 다음처럼 사용합니다.

```powershell
.venv\Scripts\Activate.ps1
```

노트북을 열 때는 VS Code에서 `.venv`의 Python 커널을 선택합니다. FireBench 자료는 공개 GCS에서 익명으로 읽지만, 네트워크 연결이 필요합니다.

## 모델 학습

프로젝트 루트에서 실행합니다.

```bash
python firespread/train_from_csv.py
```

동작 방식:

1. 루트에 `firebench_dataset.csv`가 있으면 현재 CSV 스키마를 확인해 읽습니다.
2. CSV가 없으면 FireBench에서 최대 30개 시뮬레이션 샘플을 추출해 CSV를 생성합니다.
3. 데이터를 학습·검증 세트로 나누고 다중 출력 Random Forest 회귀 모델을 학습합니다.
4. `fire_model.pkl`을 저장하고 방향·속도 각각의 MAE와 RMSE를 출력합니다.

생성 파일인 `firebench_dataset.csv`와 `fire_model.pkl`은 프로젝트 루트에 만들어집니다. CSV 스키마가 바뀌었거나 이전 버전으로 만든 CSV를 사용할 경우, 기존 CSV를 제거한 뒤 다시 생성해야 합니다.

## 확산 예측

예측 스크립트는 다음 입력 파일을 프로젝트 루트에서 찾습니다.

- `Slope_All.tif`: 좌표계 정보가 포함된 경사도 GeoTIFF
- `fire_model.pkl`: 학습 단계에서 생성한 모델
- `.env`: 기상청 API 서비스 키

`.env`는 저장소 루트에 만들고 다음 변수를 설정합니다.

```dotenv
WEATHER_API_KEY=발급받은_기상청_API_키
```

기본 예시 좌표와 파일 경로를 확인·수정한 뒤 프로젝트 루트에서 실행합니다.

```bash
python firespread/fire_predictor.py
```

파이썬 코드에서 직접 호출할 수도 있습니다.

```python
import os
import sys
from dotenv import load_dotenv

sys.path.insert(0, "firespread")
load_dotenv()

from fire_predictor import predict_fire_at

result = predict_fire_at(
    lat=37.4563,
    lon=126.7052,
    slope_axis_azimuth_deg=135.0,
    geotiff_path="Slope_All.tif",
	weather_api_key=os.environ["WEATHER_API_KEY"],
    model_path="fire_model.pkl",
)
```

`slope_axis_azimuth_deg`는 FireBench의 로컬 `+x`축을 기준으로 한 지리 방위각(북쪽 0°, 시계방향)입니다. FireBench 경로와 전역 메타데이터는 이 방위각을 제공하지 않으므로, 예제값을 실제 현장 설정으로 간주하지 말고 좌표축 정렬 정보를 별도로 확인해야 합니다.

## 입력 특징과 출력

학습용 입력 CSV 열은 다음과 같습니다.

| 열 | 의미 |
|---|---|
| `wind_dir_local` | FireBench 로컬 좌표계의 풍향(도) |
| `wind_speed` | 풍속(m/s) |
| `slope_deg` | 경사각(도) |

목표 열은 `spread_dir_local`(로컬 확산 방향, 도)과 `spread_speed`(확산 속도, m/s)입니다. 로컬 방향은 FireBench 좌표계 값이며, 지리 방위각과 같다고 볼 수 없습니다.

## 주요 한계

- 현재 파이프라인은 CCTV 이미지에서 화재를 감지하거나 픽셀 위치를 산출하지 않습니다. 확산 예측에 위경도를 직접 입력합니다.
- FireBench는 제어된 시뮬레이션 자료입니다. 한국의 실제 수종·연료 수분·습도·계절·복합 지형을 충분히 표현하지 않으므로, 실제 산불 예측 성능은 국내 관측 자료로 별도 검증해야 합니다.
- 확산 라벨은 연속 시점 사이 표면 온도 합의 최댓값 위치 이동으로 계산한 근사치입니다.
- 기상청 API는 격자 관측값을 제공합니다. 특정 산지의 국지 바람과 화재 주변 유동을 직접 측정한 값은 아닙니다.
- 예측 결과는 연구·프로토타입 용도이며, 검증된 현장 대응 판단이나 공식 산불 예보를 대체하지 않습니다.
- 노트북의 설명용 합성 예시는 실제 FireBench에서 추출한 결과가 아닙니다.

## 참고 자료

- [FireBench 데이터셋 및 연구 소개](https://sites.research.google/wildfires/firebench/)
- [FireBench 공개 저장소](https://github.com/google-research/firebench)
- [기상청 단기예보 조회서비스](https://www.data.go.kr/data/15084084/openapi.do)
