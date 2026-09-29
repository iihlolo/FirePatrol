# firebench_loader.py
# : FireBench Zarr 데이터를 로드하고 학습 샘플을 추출하여 CSV 파일 생성

import numpy as np
import pandas as pd
import xarray as xr
import re
from typing import Tuple, List
from typing import Optional

try:
    import gcsfs
    fs = gcsfs.GCSFileSystem(token="anon")  # public-read 권한의 GCS bucket
except ImportError:
    print("⚠️ gcsfs 라이브러리가 설치되지 않았습니다. 설치하려면: pip install gcsfs")
    fs = None

# -------------------------------------------------------------
# GCS 경로에서 metadata parsing 보조 함수
# -------------------------------------------------------------

def extract_slope_from_path(path: str) -> float:
    """Extract slope *steepness* in degrees from 'rampXX.X' pattern."""
    m = re.search(r"ramp([\d.]+)", path)
    return float(m.group(1)) if m else np.nan

# -------------------------------------------------------------
# Low‑level Zarr access 함수
# -------------------------------------------------------------

def _open_dataset(path: str) -> Optional[xr.Dataset]:
    if not path.startswith("gs://"):
        gcs_path = "gs://" + path
    else:
        gcs_path = path

    backend_kw = {"consolidated": True, "storage_options": {"token": "anon"}}

    try:
        return xr.open_dataset(
            gcs_path,
            engine="zarr",
            **backend_kw,
            decode_timedelta=True
        )
    except (ValueError, TypeError): # consolidated metadata 없거나 FSMap bug
        backend_kw["consolidated"] = False
        try:
            return xr.open_dataset(
                gcs_path,
                engine="zarr",
                **backend_kw,
                decode_timedelta=True
            )
        except Exception as e:
            print(f"❌ Zarr 열기 실패: {path} → {e}")
            return None

# -------------------------------------------------------------
# 단일 simulation에서 특징 추출
# -------------------------------------------------------------

def _wind_stats(ds_t: xr.Dataset) -> tuple[float, float]:
    # 메모리 사용량 절감을 위한 spatial slicing
    u = ds_t["u"].isel(x=slice(None, None, 4), y=slice(None, None, 4), z=slice(None, None, 16))
    v = ds_t["v"].isel(x=slice(None, None, 4), y=slice(None, None, 4), z=slice(None, None, 16))

    u_bar = u.mean().compute().item()
    v_bar = v.mean().compute().item()

    speed = float(np.hypot(u_bar, v_bar))
    direction = float((np.degrees(np.arctan2(v_bar, u_bar)) + 360) % 360)
    return direction, speed

def _spread_stats(ds: xr.Dataset, t_idx: int, T_thresh: float = 50) -> tuple[float, float]:
    T_now = (
        ds["T_s"]
        .isel(t=t_idx, z=slice(None, None, 16))
        .sum(dim="z")
        .transpose("y", "x")
    )
    if T_now.max().item() < T_thresh:
        return np.nan, np.nan
    ctr_now = np.unravel_index(np.argmax(T_now.values), T_now.shape)

    if t_idx + 1 >= ds.sizes["t"]:
        return 0.0, 0.0

    T_next = (
        ds["T_s"]
        .isel(t=t_idx + 1, z=slice(None, None, 16))
        .sum(dim="z")
        .transpose("y", "x")
    )
    if T_next.max().item() < T_thresh:
        return 0.0, 0.0
    ctr_next = np.unravel_index(np.argmax(T_next.values), T_next.shape)

    if "x" not in ds.coords or "y" not in ds.coords or "t" not in ds.coords:
        raise ValueError("FireBench Zarr에 x, y, t 좌표가 있어야 확산 속도를 계산할 수 있습니다.")

    x_coords = ds["x"].values
    y_coords = ds["y"].values
    dx = float(x_coords[ctr_next[1]] - x_coords[ctr_now[1]])
    dy = float(y_coords[ctr_next[0]] - y_coords[ctr_now[0]])

    time_coords = ds["t"].isel(t=slice(t_idx, t_idx + 2)).values
    time_delta = time_coords[1] - time_coords[0]
    if np.issubdtype(time_coords.dtype, np.timedelta64):
        seconds_per_step = float(time_delta / np.timedelta64(1, "s"))
    else:
        time_unit = str(ds["t"].attrs.get("units", "")).strip().lower()
        seconds_per_unit = {
            "ns": 1e-9,
            "nanosecond": 1e-9,
            "nanoseconds": 1e-9,
            "us": 1e-6,
            "microsecond": 1e-6,
            "microseconds": 1e-6,
            "ms": 1e-3,
            "millisecond": 1e-3,
            "milliseconds": 1e-3,
            "s": 1.0,
            "second": 1.0,
            "seconds": 1.0,
        }.get(time_unit)
        if seconds_per_unit is None:
            raise ValueError(f"지원하지 않는 FireBench 시간 단위입니다: {time_unit!r}")
        seconds_per_step = float(time_delta) * seconds_per_unit

    if not np.isfinite(seconds_per_step) or seconds_per_step <= 0:
        raise ValueError(f"잘못된 시간 간격입니다: {seconds_per_step}")

    speed = float(np.hypot(dx, dy) / seconds_per_step)
    direction = float((np.degrees(np.arctan2(dy, dx)) + 360) % 360)
    return direction, speed

def load_firebench_sample_from_path(path: str, time_frac: float = 0.5) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    ds = _open_dataset(path)
    if ds is None:
        return None, None

    if "T_s" not in ds.data_vars and "T_surf" in ds.data_vars:
        ds = ds.rename({"T_surf": "T_s"})

    need_vars = {"u", "v", "T_s"}
    if not need_vars.issubset(ds.data_vars):
        print(f"❌ 필요한 변수 {need_vars} 누락: {path}")
        return None, None

    if not np.isfinite(time_frac) or not 0.0 <= time_frac < 1.0:
        raise ValueError("time_frac는 0 이상 1 미만의 유한한 값이어야 합니다.")

    t_idx = int(time_frac * ds.sizes["t"])
    ds_t = ds.isel(t=t_idx, drop=True)

    wind_dir, wind_speed = _wind_stats(ds_t)
    spread_dir, spread_speed = _spread_stats(ds, t_idx)

    if np.isnan(spread_dir) or np.isnan(spread_speed):
        print(f"⚠️ spread 계산 실패 (NaN): {path}")
        return None, None

    slope_deg = extract_slope_from_path(path)
    X = np.array([[wind_dir, wind_speed, slope_deg]], dtype=np.float32)
    y = np.array([[spread_dir, spread_speed]], dtype=np.float32)
    return X, y

# -------------------------------------------------------------
# ramp directories 순회하여 sample loading
# -------------------------------------------------------------

def _ramp_dirs(prefix: str = "firebench/v2024.04") -> list[str]:
    gcs = fs
    if gcs is None:
        raise RuntimeError(
            "GCS 접근에는 gcsfs가 필요합니다. 현재 환경에 설치하세요: pip install gcsfs"
        )

    level1 = gcs.ls(prefix, detail=False)
    ramps = []
    for p in level1:
        if "_$folder$" in p:
            continue
        if re.search(r"ramp\d", p):
            ramps.append(p if p.endswith('/') else p + '/')
        else:
            sub = gcs.ls(p, detail=False)
            for q in sub:
                if "_$folder$" in q:
                    continue
                if re.search(r"ramp\d", q):
                    ramps.append(q if q.endswith('/') else q + '/')
    return ramps

def load_all_samples_from_firebench(prefix: str = "firebench/v2024.04", max_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    count = 0

    for ramp_dir in _ramp_dirs(prefix):
        if count >= max_samples:
            break
        zarr_path = f"{ramp_dir}fire.zarr"
        print(f"📦 로딩: {zarr_path}")
        X, y = load_firebench_sample_from_path(zarr_path)
        if X is not None and y is not None:
            X_all.append(X)
            y_all.append(y)
            count += 1

    if not X_all:
        raise RuntimeError("유효한 샘플을 하나도 수집하지 못했습니다.")

    return np.vstack(X_all), np.vstack(y_all)

# -------------------------------------------------------------
# sample dataset을 CSV 파일로 저장 
# -------------------------------------------------------------

def save_dataset_as_csv(X: np.ndarray, y: np.ndarray, filepath: str = "firebench_dataset.csv") -> None:
    cols = ["wind_dir_local", "wind_speed", "slope_deg", "spread_dir_local", "spread_speed"]
    df = pd.DataFrame(np.hstack([X, y]), columns=cols)
    df.to_csv(filepath, index=False)
    print(f"✅ 저장 완료: {filepath}")
def find_fire_zarrs(fs, path, max_depth=2):
  if path.endswith('fire.zarr'):
    return [path]
  if fs.info(path)['type'] == 'directory':
    if max_depth >= 0:
      out = []
      for sub_path in fs.ls(path):
        out.extend(find_fire_zarrs(fs, sub_path, max_depth - 1))
      return out
  return []

