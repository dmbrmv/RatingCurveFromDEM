"""Functions for generating river geometries from branch data.

This module contains functions for creating GeoDataFrames of river geometries
from branch features.

"""
import os

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point

from .geom_proc import round_down, round_up


def rank_to_acc(rank_row: pd.Series) -> str:
    """Convert a river rank to an approximate accumulation range.

    The accumulation range is an approximate representation of the drainage
    area of the river. The ranges are based on the following values:
    - 1e3 - 1e4: small creeks
    - 1e4 - 1e5: small rivers
    - 1e5 - 1e6: medium rivers
    - 1e6 - 1e7: rivers
    - 1e7 - 1e8: big rivers
    - 1e8 - 1e9: large rivers
    - 1e2 - 1e3: unknown/default

    Args:
        rank_row (pd.Series): A row of a GeoDataFrame containing a 'rank'
            column.

    Returns:
        str: A string representing the approximate accumulation range.

    """
    rank = rank_row["rank"]
    if rank == "big_creeks":
        return f"{1e3:.0f} - {1e4:.0f}"
    elif rank == "small_rivers":
        return f"{1e4:.0f} - {1e5:.0f}"
    elif rank == "medium_rivers":
        return f"{1e5:.0f} - {1e6:.0f}"
    elif rank == "rivers":
        return f"{1e6:.0f} - {1e7:.0f}"
    elif rank == "big_rivers":
        return f"{1e7:.0f} - {1e8:.0f}"
    elif rank == "large_rivers":
        return f"{1e8:.0f} - {1e9:.0f}"
    else:
        return f"{1e2:.0f} - {1e3:.0f}"


def get_river_geom(pseudo_rank, branches):
    """Generate a GeoDataFrame of river geometries from branch data.

    Args:
        pseudo_rank (str): A rank to assign to each river branch.
        branches (dict): A dictionary containing branch features with geometry data.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing river geometries with IDs and ranks.

    """
    # Initialize an empty GeoDataFrame with a geometry column
    temp_river = pd.DataFrame(columns=["geometry"])
    temp_river = gpd.GeoDataFrame(temp_river, geometry="geometry", crs=4326)

    # Initialize 'id' and 'rank' columns with default values
    temp_river["id"] = 0
    temp_river["rank"] = ""

    # Iterate over each branch feature and extract its geometry
    for i, branch in enumerate(branches["features"]):
        line = LineString(branch["geometry"]["coordinates"])  # Create a LineString from coordinates
        temp_river.loc[i, "rank"] = pseudo_rank  # Assign the pseudo rank
        temp_river.loc[i, "geometry"] = line  # Store the geometry in the GeoDataFrame
        temp_river.loc[i, "id"] = int(branch["id"])  # Assign the branch ID

    return temp_river


def distance_to_line(line: LineString, point: Point) -> float:
    """Calculate the distance from a point to a line.

    Args:
        line (LineString): The line from which the distance is calculated.
        point (Point): The point from which the distance is calculated.

    Returns:
        float: The distance from the point to the line.

    """
    return line.distance(point)


def river_points(river_geom: gpd.GeoDataFrame):
    """Extract points from the geometry of the first river feature in the GeoDataFrame.

    Args:
        river_geom (gpd.GeoDataFrame): A GeoDataFrame containing river geometries.

    Returns:
        List[Point]: A list of shapely Points representing the coordinates of the river geometry.

    """
    # Extract the y and x coordinates from the first geometry
    y_coords, x_coords = river_geom.loc[0, "geometry"].xy

    # Create a list of Points from the extracted coordinates
    return [Point(y, x) for x, y in zip(x_coords, y_coords)]


def get_river_points(pnt, partial_path):
    """Retrieve the file path of the river point geospatial data if it exists.

    This function calculates the longitude and latitude values rounded to the
    nearest 0.5, constructs a filename based on these values, and checks if
    the corresponding file exists in the specified directory structure.

    Args:
    ----
        pnt: The point object containing x (longitude) and y (latitude) attributes.
        partial_path: The base path to the directory containing partial data.

    Returns:
    -------
        str or None: The file path if the file exists, otherwise None.

    """
    # Extract longitude and latitude from the point object
    lon = pnt.x
    lat = pnt.y

    # Round the longitude and latitude to the nearest 0.5
    lon_txt = round_down(lon, round_val=0.5)
    lat_txt = round_up(lat, round_val=0.5)

    # Get a list of subfolder paths in the partial directory
    subfolders = [f.path for f in os.scandir(f"{partial_path}/partial/") if f.is_dir()]

    # Iterate through each subfolder to find the file
    for subfolder in subfolders:
        tag = os.path.basename(subfolder)
        filename = f"{partial_path}/partial/{tag}/{tag}_{lon_txt}_{lat_txt}.gpkg"

        # Check if the file exists; return the filename if it does
        if os.path.exists(filename):
            return filename

    # Return None if no file was found
    return None
