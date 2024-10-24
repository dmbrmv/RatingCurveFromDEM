from shapely.geometry import MultiPolygon, Polygon
from typing import Union
import numpy as np
import geopandas as gpd
import math
import ast
from functools import reduce
from numba import jit
from numpy import arctan2, cos, sin, sqrt, radians, pi, append, diff
import os


def area_from_gdf(poly):
    """Calculate area of shape stored in GeoDataFrame.

    Args:
    ----
        poly (GeoDataFrame): Desired shape

    Returns:
    -------
        area (float): area of object in sq. km

    """
    if poly.empty:
        return np.NaN
    else:
        poly = poly_from_multipoly(poly["geometry"][0])
        area = polygon_area(poly)
    return area


def find_float_len(number: float) -> int:
    return len(str(number).split(".")[1]) >= 2


def min_max_xy(shp_file):
    x_max, x_min = np.max(shp_file.exterior.xy[0]), np.min(shp_file.exterior.xy[0])

    y_max, y_min = np.max(shp_file.exterior.xy[1]), np.min(shp_file.exterior.xy[1])

    return (x_min, y_min, x_max, y_max)


def polygon_area(geo_shape, radius=6378137):
    """Compute area of spherical polygon, assuming spherical Earth.

    Returns result in ratio of the sphere's area if the radius is specified.
    Otherwise, in the units of provided radius.
    lats and lons are in degrees.

    """
    lons, lats = geo_shape.exterior.xy
    lats, lons = np.deg2rad(lats), np.deg2rad(lons)
    # Line integral based on Green's Theorem, assumes spherical Earth

    # close polygon
    if lats[0] != lats[-1]:
        lats = append(lats, lats[0])
        lons = append(lons, lons[0])

    # colatitudes relative to (0,0)
    a = sin(lats / 2) ** 2 + cos(lats) * sin(lons / 2) ** 2
    colat = 2 * arctan2(sqrt(a), sqrt(1 - a))

    # azimuths relative to (0,0)
    az = arctan2(cos(lats) * sin(lons), sin(lats)) % (2 * pi)

    # Calculate diffs
    # daz = diff(az) % (2*pi)
    daz = diff(az)
    daz = (daz + pi) % (2 * pi) - pi

    deltas = diff(colat) / 2
    colat = colat[0:-1] + deltas

    # Perform integral
    integrands = (1 - cos(colat)) * daz

    # Integrate
    area = abs(sum(integrands)) / (4 * pi)

    area = min(area, 1 - area)
    if radius is not None:  # return in units of radius
        return area * 4 * pi * radius**2 / 10**6
    else:  # return in ratio of sphere total area
        return area / 10**6


def poly_from_multipoly(ws_geom):
    """Return biggest polygon from multipolygon WS.

    It's the real WS, and not malfunctioned part of it

    """
    if isinstance(ws_geom, MultiPolygon):
        big_area = [polygon_area(geo_shape=polygon) for polygon in ws_geom.geoms]
        ws_geom = ws_geom.geoms[np.argmax(big_area)]
    else:
        ws_geom = ws_geom
    return ws_geom


def ws_AOI(ws, shp_path: str):
    def my_ceil(a, precision=0):
        return np.true_divide(np.ceil(a * 10**precision), 10**precision)

    def my_floor(a, precision=0):
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    LONS, LATS = ws.exterior.xy
    max_LAT = np.max(LATS)
    max_LON = np.max(LONS)
    min_LAT = np.min(LATS)
    min_LON = np.min(LONS)

    min_LON, max_LON, min_LAT, max_LAT = (
        my_floor(min_LON, 2),
        my_ceil(max_LON, 2),
        my_floor(min_LAT, 2),
        my_ceil(max_LAT, 2),
    )

    test = gpd.GeoDataFrame()
    aoi_geom = Polygon([[min_LON, min_LAT], [min_LON, max_LAT], [max_LON, max_LAT], [max_LON, min_LAT]])
    test.loc[0, "geometry"] = aoi_geom  # type: ignore
    test = test.set_crs(epsg=4326)

    test.to_file(f"{shp_path}", index=False)

    return shp_path


def round_up(x, round_val: float = 5):
    return int(np.ceil(x / round_val)) * round_val


def round_down(x, round_val: float = 5):
    return int(np.floor(x / round_val)) * round_val


def find_extent(ws: Polygon, grid_res: float, dataset: str = ""):
    """_summary_.

    Args:
    ----
        ws (Polygon): _description_
        grid_res (float): _description_

    """

    def x_round(x):
        return round((x - 0.25) * 2) / 2 + 0.25

    def round_nearest(x, a):
        max_frac_digits = 10
        for i in range(max_frac_digits):
            if round(a, -int(math.floor(math.log10(a))) + i) == a:
                frac_digits = -int(math.floor(math.log10(a))) + i
                break
        return round(round(x / a) * a, frac_digits)  # type: ignore

    lons, lats = ws.exterior.xy  # type: ignore
    max_LAT = max(lats)
    max_LON = max(lons)
    min_LAT = min(lats)
    min_LON = min(lons)

    if dataset == "gpcp":
        return [x_round(min_LON), x_round(max_LON), x_round(min_LAT), x_round(max_LAT)]
    elif bool(dataset):
        return [
            round_nearest(min_LON, grid_res),
            round_nearest(max_LON, grid_res),
            round_nearest(min_LAT, grid_res),
            round_nearest(max_LAT, grid_res),
        ]
    else:
        raise Exception(f"Something wrong ! {dataset} -- {grid_res}")


def create_gdf(shape: Union[Polygon, MultiPolygon]):
    """Create geodataframe with given shape as a geometry."""
    gdf_your_WS = poly_from_multipoly(ws_geom=shape)
    # WS from your data
    gdf_your_WS = gpd.GeoSeries([gdf_your_WS])

    # Create extra gdf to use geopandas functions
    gdf_your_WS = gpd.GeoDataFrame({"geometry": gdf_your_WS})
    gdf_your_WS = gdf_your_WS.set_crs("EPSG:4326")

    return gdf_your_WS


def RotM(alpha):
    """Rotation Matrix for angle ``alpha``."""
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([[ca, -sa], [sa, ca]])


def getSquareVertices(mm, h, phi):
    """Calculate the for vertices for square with center ``mm``, side length ``h`` and rotation ``phi``."""
    hh0 = np.ones(2) * h  # initial corner
    vv = [
        np.asarray(mm) + reduce(np.dot, [RotM(phi), RotM(np.pi / 2 * c), hh0]) for c in range(4)
    ]  # rotate initial corner four times by 90°
    return np.asarray(vv)


@jit(nopython=True)
def point_distance(new_lon, new_lat, old_lon, old_lat):
    dlon = old_lon - new_lon
    dlat = old_lat - new_lat

    a = sin(dlat / 2) ** 2 + cos(old_lat) * cos(new_lat) * sin(dlon / 2) ** 2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))

    return 6373.0 * c


def inside_mask(pt, bnd):
    x1, y1, x2, y2 = bnd
    x, y = pt
    if x1 < x and x < x2:
        if y1 < y and y < y2:
            return True
    return False


def get_river_points(pnt, partial_path):
    lon = pnt.x
    lat = pnt.y
    # pt = [lon, lat]

    lon_txt = round_down(lon, round_val=0.5)
    lat_txt = round_up(lat, round_val=0.5)

    # tiles = dict()
    subfolders = [f.path for f in os.scandir(f"{partial_path}/partial/") if f.is_dir()]

    for subfolder in subfolders:
        tag = os.path.basename(subfolder)
        filename = f"{partial_path}/partial/{tag}/{tag}_{lon_txt}_{lat_txt}.gpkg"
        if os.path.exists(filename):
            return filename

        # if inside_mask(pt, bnd):
        #     print(f"{pt} in {tag} with bnd: {bnd}")
        #     return f'{partial_path}/partial/{tag}/{tag}_{lon_txt}_{lat_txt}.gpkg'
    return None


def str_to_np(x):
    x = " ".join(x.split())

    return np.array(ast.literal_eval(x.replace(" ", ",")))


def update_geometry(pnt, tile):
    scale_actual = {
        "small_rivers": 80,
        "medium_rivers": 800,
        "rivers": 8000,
        "big_rivers": 80000,
        "large_rivers": 800000,
    }
    point_in_tile = gpd.read_file(tile)

    old_lon = pnt.x
    old_lat = pnt.y

    point_in_tile["geom_np"] = point_in_tile["geom_np"].apply(lambda x: str_to_np(x))

    point_in_tile["distance"] = point_in_tile["geom_np"].apply(
        lambda x: point_distance(
            new_lon=radians(x[0]),
            new_lat=radians(x[1]),
            old_lon=radians(old_lon),
            old_lat=radians(old_lat),
        )
    )

    point_in_tile["approx_area"] = point_in_tile["rank"].apply(lambda x: scale_actual[x])

    res_tile = (
        point_in_tile.sort_values(by=["distance", "approx_area"], axis=0)
        .reset_index(drop=True)
        .loc[[0], ["geometry", "distance", "approx_area"]]
    )
    return res_tile
