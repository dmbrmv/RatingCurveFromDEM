"""Create tiff file of flood zone for predicted level on gauge."""

import pathlib
from typing import Optional, Tuple, Union
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from shapely.geometry import Point
import numpy as np
import rasterio
from pysheds.grid import Grid
from pysheds.view import Raster


def gauge_to_utm(
    gauge_series: gpd.GeoSeries, return_gdf: bool = False
) -> Optional[Union[Tuple[gpd.GeoSeries, int], Point]]:
    """Convert geometry projected file to it metric projection.

    Args:
    ----
        gauge_series (gpd.GeoSeries):
            GeoDataFrame file with geometry in WGS-84

        return_gdf (bool, optional):
            If require just coordinates we return projected Point object.
            Otherwise GeoDataFrame, and EPSG code for further trasnformations.
            Defaults to False.

    Returns:
    -------
        Optional[ Union[Tuple[gpd.GeoSeries, int], Point]]

    """
    gdf_file = gpd.GeoSeries(gauge_series, crs=4326)
    gdf_file = gpd.GeoDataFrame(gdf_file, columns=["geometry"])
    tiff_epsg = gdf_file.estimate_utm_crs().to_epsg()
    gdf_file["x"] = gdf_file["geometry"].x
    gdf_file["y"] = gdf_file["geometry"].y
    gdf_file_crs = gdf_file.to_crs(tiff_epsg)
    if return_gdf:
        return (gdf_file, tiff_epsg)
    return gdf_file_crs["geometry"].values[0]


def gauge_buffer_creator(
    gauge_geometry: Point, ws_gdf: gpd.GeoSeries, tiff_epsg: int
) -> Tuple[gpd.GeoDataFrame, Tuple[float, float, float, float], float, float]:
    """Create squared buffer for extent of flood modelling.

    Args:
    ----
        gauge_geometry (Point): Shapely Point object from geometry column
        ws_gdf (gpd.GeoSeries): Watershed-object based on gauge_id GeoDataFrame
        tiff_epsg (int): Metric EPSG code

    Returns:
    -------
        * Trimmed buffer around point of interest
        * extent coordinates of full-sized buffer
        * number of cells which will be a limit value for innudation mapping
        * watershed area in sq. km

        Tuple[gpd.GeoDataFrame, Tuple[float, float, float, float], float, float]

    """
    # area from gauge. Calculate it in case we do not have it precalculated
    ws_area = ws_gdf.to_crs(epsg=tiff_epsg).area.values[0] * 1e-6

    # buffer size; so square will have side equal x2 of buffer size
    if 500000 < ws_area:
        # degrees
        AREA_SIZE = 0.30
    elif 50000 < ws_area <= 500000:
        # degrees
        AREA_SIZE = 0.20
    elif 5000 < ws_area <= 50000:
        # degrees
        AREA_SIZE = 0.10
    else:
        # degrees
        AREA_SIZE = 0.05
    acc_coef = (ws_area * 1e6) / (90 * 90)
    # create buffer for point
    buffer = gauge_geometry.buffer(AREA_SIZE, cap_style="square")
    # create buffer for river intersection search
    buffer_isc = gauge_geometry.buffer(AREA_SIZE - 0.015, cap_style="square")
    buffer_gdf = pd.DataFrame(
        columns=["geometry"],
    )
    #    index=[0])
    buffer_gdf = gpd.GeoDataFrame(buffer_gdf, geometry="geometry", crs=4326)  # type: ignore
    buffer_gdf.loc[0, "geometry"] = buffer_isc
    # !TODO find out does this transformation is neccessary
    # buffer_gdf = buffer_gdf.to_crs(epsg=tiff_epsg)

    # get coords from buffer
    upper_left_x, lower_right_y, lower_right_x, upper_left_y = buffer.bounds

    wgs_window = (upper_left_x, upper_left_y, lower_right_x, lower_right_y)

    return (buffer_gdf, wgs_window, acc_coef, ws_area)


def gdal_extent_clipper(
    initial_tiff: str, extent: tuple, tmp_tiff: str, final_tiff: str, crs_epsg: int
) -> None:
    """Clip and reproject .tiff file for desired extent and epsg.

    Args:
    ----
        initial_tiff (str): Ininitial .tiff file which need to be trimmed
        extent (tuple): Min, max coordinates for area of interest
        tmp_tiff (str): Name for file in temporary folder (trimmed, not projected)
        final_tiff (str): Name for projected and trimmed file
        crs_epsg (int): EPSG code

    Returns:
    -------
        None

    """
    # clip big tiff for extent
    merged_tiff = gdal.Translate(destName=tmp_tiff, srcDS=initial_tiff, projWin=extent)
    merged_tiff.FlushCache()
    # reproject for desired CRS
    merged_tiff_proj = gdal.Warp(
        destNameOrDestDS=final_tiff,
        format="GTiff",
        dstNodata=None,
        srcDSOrSrcDSTab=tmp_tiff,
        dstSRS=f"EPSG:{crs_epsg}",
    )
    merged_tiff_proj.FlushCache()

    return None


def fill_dem(dem_tiff: str, elv_fill: str) -> None:
    """Fill raster elevation pits and depressions.

    Args:
    ----
        dem_tiff (str): Path to initial dem file (.tiff)
        elv_fill (str): Path to filled dem file (.tiff)

    Returns:
    -------
        None

    """
    # Read raw DEM
    grid = Grid.from_raster(dem_tiff)
    dem = grid.read_raster(dem_tiff)
    # Fill pits
    pit_filled_dem = grid.fill_pits(dem)
    # Fill depressions
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    # Resolve flats
    inflated_dem = grid.resolve_flats(flooded_dem)
    inflated_dem[inflated_dem < 0] = np.nan  # type: ignore
    grid.to_raster(file_name=elv_fill, data=inflated_dem)

    return None


def innudation_raster(
    gauge_id: str,
    temp_res_folder: Union[pathlib.Path, str],
    acc_coef: float,
    predicted_lvl: float,
    event_date: str,
    final_res_folder: Union[pathlib.Path, str],
) -> None:
    """Create innudation map (.tiff) for predicted level on ceratin date.

    Args:
    ----
        gauge_id (str): ID for gauge
        temp_res_folder (Union[pathlib.Path, str]): Temporary folder with .tiff
        acc_coef (float): Coefficient depends on ws_area for HAND mask
        predicted_lvl (float): Monthly maximum predicted level
        event_date (str): Month of prediction in "YYYY-MM" format
        final_res_folder (Union[pathlib.Path, str]): Disk storage for innudation map in .tiff

    Returns:
    -------
        None

    """
    # Instantiate grid from raster
    grid = Grid.from_raster(f"{temp_res_folder}/{gauge_id}_fill.tiff")
    dem = grid.read_raster(f"{temp_res_folder}/{gauge_id}_fill.tiff")
    acc = grid.read_raster(f"{temp_res_folder}/{gauge_id}_acc.tiff")
    fdir = grid.read_raster(f"{temp_res_folder}/{gauge_id}_dir.tiff")
    # Compute height above nearest drainage
    hand = grid.compute_hand(fdir, dem, acc > acc_coef)
    # Create a view of HAND in the catchment
    hand_view = grid.view(hand, nodata=np.nan)
    # select values based on predicted level
    inundation_extent = np.where(hand_view < predicted_lvl, predicted_lvl - hand_view, np.nan)
    # create inundation raster based on predicted lvl
    inundation = Raster(inundation_extent, grid.viewfinder)
    file_storage = pathlib.Path(f"{final_res_folder}/{gauge_id}")
    file_storage.mkdir(exist_ok=True, parents=True)
    grid.to_raster(file_name=f"{file_storage}/{event_date}_depth.tiff", data=inundation)

    return None


def find_posts_coords(post_num: int, config: dict, conversion: bool = False):
    """Find post coordinates.

    Arguments:
    ---------
        post_num: int post number
        config: dict with confis. Keys in usage is: geom_path

    Returns:
    -------
        coordinates of the post

    """
    data = gpd.read_file(config["geom_path"])
    if conversion:
        point = gauge_to_utm(gdf_file=data[data["gauge_id"] == str(post_num)])
    else:
        point = data[data["gauge_id"] == str(post_num)][["geometry"]].values[0][0]
    return point.y, point.x  # lat lon


def coords_to_xy(dem_path, coords):
    """Translate coords to point on tiff float in fractions."""
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    lat_min, lat_max = (
        np.min(dem.coords[:, 0]),
        np.max(dem.coords[:, 0]),
    )  # perhaps not the best way??
    lon_min, lon_max = np.min(dem.coords[:, 1]), np.max(dem.coords[:, 1])
    x = (coords[0] - lat_min) / (lat_max - lat_min)
    y = (coords[1] - lon_min) / (lon_max - lon_min)
    if x < 0 or y < 0 or x > 1 or y > 1:
        return 0.5, 0.5
    return x, y


def find_posts_inside(area_box: list, config: dict):
    """Find post inside area box.

    Arguments:
    ---------
        area_box: bounding box for tif file, coords in this order:  longitude_min, latitude_min, longitude_max, latitude_max
        config: dict with confis. Keys in usage is: geom_path

    Returns:
    -------
        list[tuple[int, float, float]] -- posts inside area, denoted by code, latitude, longitude

    """
    data = gpd.read_file(config["geom_path"])
    ans = []
    for row in data[["code", "latitude", "longitude"]].itertuples():
        if area_box[1] <= row[2] <= area_box[3] and area_box[0] <= row[3] <= area_box[2]:
            ans.append((row[1], row[2], row[3]))

    return ans


def find_point_elevation(dem, coords: list):
    """Find elevation of the point.

    Arguments:
    ---------
        dem: digital elevation mask
        coords: latitude, longitude

    Returns:
    -------
        elevation for post

    """
    assert len(coords) == 2, "It is not coordinates"
    area_box = dem.bbox
    assert (
        area_box[1] <= coords[0] <= area_box[3] and area_box[0] <= coords[1] <= area_box[2]
    ), "Not inside area"

    distances = list(map(lambda x: distance(coords, x), dem.coords))
    height = dem.reshape(-1)[np.argmin(distances)]

    return height
