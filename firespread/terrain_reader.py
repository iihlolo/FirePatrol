# terrain_reader.py
# : GeoTIFF 파일에서 경사도 추출 

import rasterio
from pyproj import Transformer

def get_slope_from_geotiff(filepath: str, lat: float, lon: float) -> float:

    with rasterio.open(filepath) as src:
        if src.crs is None:
            raise ValueError("GeoTIFF 파일에 좌표계(CRS) 정보가 없습니다.")

        # 위경도 → 파일 좌표계로 변환 (WGS84 → 파일 CRS)
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        row, col = src.index(x, y)

        slope_array = src.read(1)

        # 범위 검사
        if row < 0 or row >= slope_array.shape[0] or col < 0 or col >= slope_array.shape[1]:
            raise IndexError("지정한 좌표가 GeoTIFF 범위 밖에 있습니다.")

        return float(slope_array[row, col])