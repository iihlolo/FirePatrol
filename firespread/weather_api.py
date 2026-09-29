# weather_api.py
# : 기상청 API로 특정 위치의 실시간 풍향/풍속 조회

import requests
import datetime
import math
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

# 위도/경도 → 기상청 격자 변환
def latlon_to_xy(lat: float, lon: float) -> tuple[int, int]:
    RE = 6371.00877 # 지구 반경(km)
    GRID = 5.0      # 격자 간격(km)
    SLAT1 = 30.0
    SLAT2 = 60.0
    OLON = 126.0
    OLAT = 38.0
    XO = 43
    YO = 136

    DEGRAD = math.pi / 180.0
    re = RE / GRID
    slat1 = SLAT1 * DEGRAD
    slat2 = SLAT2 * DEGRAD
    olon = OLON * DEGRAD
    olat = OLAT * DEGRAD

    sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
    sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
    sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
    sf = (sf ** sn * math.cos(slat1)) / sn
    ro = math.tan(math.pi * 0.25 + olat * 0.5)
    ro = re * sf / (ro ** sn)

    ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
    ra = re * sf / (ra ** sn)
    theta = lon * DEGRAD - olon
    if theta > math.pi:
        theta -= 2.0 * math.pi
    if theta < -math.pi:
        theta += 2.0 * math.pi
    theta *= sn

    x = int(ra * math.sin(theta) + XO + 0.5)
    y = int(ro - ra * math.cos(theta) + YO + 0.5)

    return x, y

# 풍향/풍속 조회
def get_wind_data(lat: float, lon: float, service_key: str) -> tuple[float, float]:
    now_kst = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
    base_datetime = (now_kst - datetime.timedelta(minutes=45)).replace(
        minute=0, second=0, microsecond=0
    )
    base_date = base_datetime.strftime("%Y%m%d")
    base_time = base_datetime.strftime("%H%M")

    x, y = latlon_to_xy(lat, lon)

    url = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getUltraSrtNcst"
    params = {
        "serviceKey": service_key,
        "numOfRows": "100",
        "pageNo": "1",
        "dataType": "XML",
        "base_date": base_date,
        "base_time": base_time,
        "nx": x,
        "ny": y,
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    result_code = root.findtext(".//header/resultCode")
    if result_code != "00":
        result_message = root.findtext(".//header/resultMsg") or "응답 메시지 없음"
        raise RuntimeError(
            f"기상청 API 오류 (코드 {result_code or '없음'}): {result_message}"
        )
    items = root.findall(".//item")

    uuu = vvv = wspd = None

    for item in items:
        category = item.findtext("category")
        value = item.findtext("obsrValue")
        if category is None or value is None:
            continue

        if category == "UUU":
            uuu = float(value)
        elif category == "VVV":
            vvv = float(value)
        elif category == "WSD":
            wspd = float(value)

    if uuu is None or vvv is None or wspd is None:
        raise ValueError("풍향/풍속 데이터를 찾을 수 없습니다.")

    # 풍향 계산 (0° = 북, 시계방향)
    wind_dir_rad = math.atan2(vvv, uuu)
    wind_deg = (270 - math.degrees(wind_dir_rad)) % 360

    return wind_deg, wspd