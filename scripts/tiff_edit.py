"""Create tiff file of flood zone for predicted level on gauge."""

from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from pysheds.grid import Grid
from pysheds.view import Raster
from shapely.geometry import Point

gpd.options.io_engine = "pyogrio"


def gauge_to_utm(
    gauge_series: gpd.GeoSeries, return_gdf: bool = False
) -> Optional[Union[Tuple[gpd.GeoSeries, int], Point]]:
    """Convert geometry projected file to metric projection.

    Args:
    ----
        gauge_series (gpd.GeoSeries):
            GeoSeries containing geometry in WGS-84.

        return_gdf (bool, optional):
            If True, return a GeoDataFrame with the projected geometry and EPSG code.
            If False, return a Point object with the projected coordinates.
            Defaults to False.

    Returns:
    -------
        Optional[ Union[Tuple[gpd.GeoSeries, int], Point]]:
            If return_gdf is True, returns a tuple containing the GeoDataFrame and EPSG code.
            If return_gdf is False, returns a projected Point object.

    """
    # Create GeoDataFrame from GeoSeries
    gdf_file = gpd.GeoSeries(gauge_series, crs=4326)
    gdf_file = gpd.GeoDataFrame(gdf_file, columns=["geometry"])

    # Estimate UTM CRS and get EPSG code
    tiff_epsg = gdf_file.estimate_utm_crs().to_epsg()

    # Extract x and y coordinates
    gdf_file["x"] = gdf_file["geometry"].x
    gdf_file["y"] = gdf_file["geometry"].y

    # Project to UTM CRS
    gdf_file_crs = gdf_file.to_crs(tiff_epsg)

    if return_gdf:
        return (gdf_file, tiff_epsg)

    return gdf_file_crs["geometry"].values[0]


def gauge_buffer_creator(gauge_geometry: Point):
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
    AREA_SIZE = 0.5
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
    # get coords from buffer
    upper_left_x, lower_right_y, lower_right_x, upper_left_y = buffer.bounds

    wgs_window = (upper_left_x, upper_left_y, lower_right_x, lower_right_y)

    return (buffer_gdf, wgs_window)


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
    inflated_dem[inflated_dem < 0] = 0  # type: ignore
    grid.to_raster(file_name=elv_fill, data=inflated_dem)

    return None


def truncate(value: float, digits: int) -> float:
    """Truncate a given value to the specified number of decimal places.

    The truncation is done by multiplying the value by a scaling factor, rounding,
    and then dividing by the scaling factor.

    Args:
    ----
        value (float): The value to be truncated.
        digits (int): The number of decimal places to truncate to.

    Returns:
    -------
        float: The truncated value.

    """
    factor = 10**digits
    return np.round((value * factor) / factor, 2)


def conditional_truncate(value):
    """Apply conditions to determine truncation digits.

    The truncation is determined by the value itself:
    - If value is less than 1, truncate to 2 decimal places.
    - If value is greater than or equal to 1 and less than 5, truncate to 1 decimal place.
    - If value is greater than or equal to 5, truncate to 0 decimal places.

    Args:
        value (float): The value to truncate.

    Returns:
        float: The truncated value.

    """
    if value < 1:
        lvl_info = truncate(value, 2)
    elif value > 1 and value < 5:
        lvl_info = truncate(value, 1)
    else:
        lvl_info = truncate(value, 0)
    return lvl_info


def innudation_raster(
    gauge_id: str,
    temp_res_folder: Union[Path, str],
    acc_coef: float,
    predicted_lvl: float,
    event_date: str,
    final_res_folder: Union[Path, str],
) -> None:
    """Create inundation map (.tiff) for predicted level on a certain date.

    Args:
    ----
        gauge_id (str): ID for gauge
        temp_res_folder (Union[Path, str]): Temporary folder with .tiff
        acc_coef (float): Coefficient depends on ws_area for HAND mask
        predicted_lvl (float): Monthly maximum predicted level
        event_date (str): Month of prediction in "YYYY-MM" format
        final_res_folder (Union[Path, str]): Disk storage for inundation map in .tiff

    Returns:
    -------
        None

    """
    # Instantiate grid from raster
    grid = Grid.from_raster(f"{temp_res_folder}/{gauge_id}_fill_poly.tiff", data_name="fill_grid")
    
    # Read DEM, accumulation, and flow direction data from respective files
    dem = grid.read_raster(f"{temp_res_folder}/{gauge_id}_fill_poly.tiff", data_name="fill_dem")
    acc = grid.read_raster(f"{temp_res_folder}/{gauge_id}_acc_poly.tiff", data_name="acc")
    fdir = grid.read_raster(f"{temp_res_folder}/{gauge_id}_dir_poly.tiff", data_name="fdir")
    
    # Compute height above nearest drainage
    hand = grid.compute_hand(fdir, dem, acc > acc_coef)
    
    # Create a view of HAND in the catchment
    hand_view = grid.view(hand, nodata=np.nan)
    
    # Calculate inundation extent based on predicted level
    inundation_extent = np.where(hand_view < predicted_lvl, predicted_lvl - hand_view, np.nan)
    
    # Create inundation raster
    inundation = Raster(inundation_extent, grid.viewfinder)
    
    # Ensure the output directory exists
    file_storage = Path(f"{final_res_folder}/{gauge_id}")
    file_storage.mkdir(exist_ok=True, parents=True)
    
    # Determine the level information for file naming
    lvl_info = conditional_truncate(value=predicted_lvl)
    
    # Save the inundation data as a raster file
    grid.to_raster(
        file_name=f"{file_storage}/{event_date}_{lvl_info}_depth.tiff",
        data=inundation,
        data_name="innudation",
        nodata_out=np.nan,
    )

    return None


def find_posts_coords(post_num: int, config: dict, conversion: bool = False) -> Tuple[float, float]:
    """Find coordinates of the post.

    Args:
    ----
        post_num (int): Post number.
        config (dict): Configuration dictionary. Keys in usage is: geom_path
        conversion (bool, optional): If True, convert coordinates to UTM projection. Defaults to False.

    Returns:
    -------
        coordinates of the post as (latitude, longitude)

    """
    data = gpd.read_file(config["geom_path"])
    if conversion:
        point = gauge_to_utm(gauge_series=data[data["gauge_id"] == str(post_num)])
    else:
        point = data[data["gauge_id"] == str(post_num)][["geometry"]].values[0][0]
    return point.y, point.x  # lat lon


def coords_to_xy(dem_path, coords):
    """Translate coords to point on tiff float in fractions."""
    grid = Grid.from_raster(dem_path, nodata=0)
    dem = grid.read_raster(dem_path, nodata=0)

    lat_min, lat_max = (
        np.min(dem.coords[:, 0]),
        np.max(dem.coords[:, 0]),
    )  # perhaps not the best way??
    lon_min, lon_max = np.min(dem.coords[:, 1]), np.max(dem.coords[:, 1])
    x = (coords[0] - lat_min) / (lat_max - lat_min)
    y = (coords[1] - lon_min) / (lon_max - lon_min)
    if x < 0 or y < 0 or x > 1 or y > 1:
        logger.warning(
            f"incorrect coordinates: {coords} , {lat_min}, {lat_max}, {lon_min}, {lon_max}, {x}, {y}. Return middle point"
        )
        return 0.5, 0.5
    return x, y


def load_elevation_map(filename):
    """Load and return a digital elevation map from a given file.

    This function reads a TIFF file containing a digital elevation map (DEM)
    and returns both the grid and the elevation data.

    Parameters
    ----------
    filename : str
        The path and name of the TIFF file containing the DEM.

    Returns
    -------
    grid : pysheds.grid.Grid
        An object containing information about the digital elevation map.
    dem : numpy.ndarray
        A 2D array representing the elevation data.

    """
    # Create a grid from the raster file, specifying no-data value and data name
    grid = Grid.from_raster(filename, nodata=0, data_name="dem_grid")
    # Read the elevation data from the grid
    dem = grid.read_raster(filename, nodata=0, data_name="dem")

    return grid, dem


def coords_to_numpy_num(x, y, grid):
    """Return indexes of numpy geo mask according to point coordinates.

    This function takes in point coordinates and a Grid object from the pysheds library.
    It calculates the line and column of the numpy geo mask that corresponds to the point.
    This is done by using the affine transformation from the grid to map the point coordinates
    to their corresponding indices in the numpy array.

    Parameters
    ----------
    x, y: float, float
        Point coordinates
    grid: pysheds.sgrid.sGrid
        Contains information about the digital elevation map

    Returns
    -------
    line, column: int, int
        Indexes of array

    """
    table_size = grid.shape
    # Get the step size and origin of the grid
    x_step, y_step = grid.affine[0], grid.affine[4]
    x_0, y_0 = grid.affine[2], grid.affine[5]

    # Iterate over the columns to find the correct one
    for column in range(table_size[1]):
        # Check if the point is within the current column
        if x_0 + column * x_step < x <= x_0 + column * x_step + x_step:
            break

    # Iterate over the lines to find the correct one
    for line in range(table_size[0]):
        # Check if the point is within the current line
        if y_0 + line * y_step + y_step < y <= y_0 + line * y_step:
            break

    # Return the line and column
    return line, column


def get_post_height_from_dem(pt, tiff_path: str) -> float:
    """Retrieve the height of a post from a digital elevation model (DEM).

    This function takes a geographical point and a path to a DEM file, and
    returns the elevation value at that point.

    Parameters
    ----------
    pt : Point
        The geographical point representing the location of the post.
    tiff_path : str
        Path to the DEM file.

    Returns
    -------
    float
        The elevation value at the specified point.

    """
    # Load the digital elevation map and its grid
    grid, dem = load_elevation_map(tiff_path)

    # Convert the point coordinates to grid indices
    # This is done by using the affine transformation from the grid to map the
    # point coordinates to their corresponding indices in the numpy array.
    u, v = coords_to_numpy_num(pt.x, pt.y, grid)

    # Return the elevation at the specified grid location
    return dem[u, v]
