"""Module provides functions for processing geometric data."""

import ast
import math
from functools import reduce
from typing import List, Union

import geopandas as gpd
import numpy as np
from numba import jit
from numpy import append, arctan2, cos, diff, pi, radians, sin, sqrt
from pandas import DataFrame
from shapely.geometry import MultiPolygon, Polygon


def area_from_gdf(poly: gpd.GeoDataFrame) -> float:
    """Calculate area of shape stored in GeoDataFrame.

    Args:
    ----
        poly (GeoDataFrame): Desired shape

    Returns:
    -------
        area (float): area of object in sq. km

    """
    # Check if GeoDataFrame is empty
    if poly.empty:
        return np.NaN
    else:
        # Get the first geometry from the GeoDataFrame and pass it to the
        # polygon_from_multipoly function, which will return the biggest polygon
        # in case of a multipolygon
        poly = poly_from_multipoly(poly["geometry"][0])
        # Calculate area of the polygon in sq. km
        area = polygon_area(poly)
    return area


def find_float_len(number: float) -> int:
    """Check if a given number is a float with more than one decimal place.

    Args:
    ----
        number (float): The number to check.

    Returns:
    -------
        int: 1 if the number has more than one decimal place, 0 otherwise.

    """
    # Check if the number is a float with more than one decimal place
    # by splitting the string representation of the number at the decimal
    # point and checking if the length of the second part is greater than 1
    return len(str(number).split(".")[1]) > 1


def min_max_xy(shp_file):
    """Calculate the minimum and maximum of x and y coordinates from a GeoDataFrame.

    Parameters
    ----------
    shp_file : GeoDataFrame
        The GeoDataFrame with the desired shape.

    Returns
    -------
    tuple
        A tuple with the minimum and maximum of x and y coordinates.

    """
    # Get the x (longitude) and y (latitude) coordinates from the GeoDataFrame
    x, y = shp_file.exterior.xy

    # Calculate the minimum and maximum of x and y coordinates
    x_max, x_min = np.max(x), np.min(x)
    y_max, y_min = np.max(y), np.min(y)

    # Return the minimum and maximum of x and y coordinates as a tuple
    return (x_min, y_min, x_max, y_max)


def polygon_area(geo_shape, radius=6378137):
    """Compute area of spherical polygon, assuming spherical Earth.

    This function computes the area of a polygon on the surface of a sphere,
    given the latitude and longitude coordinates of its vertices. The area is
    computed using the Gaussian quadrature method.

    Parameters
    ----------
    geo_shape : shapely.geometry.Polygon
        The polygon for which to compute the area.
    radius : float, optional
        The radius of the sphere. If not specified, the area is returned in
        ratio of the sphere's area. Otherwise, in the units of provided radius.

    Returns
    -------
    float
        The area of the polygon in square kilometers.

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
    """Return the biggest polygon from a multipolygon WS.

    The WS is a water body, and it's possible that the geometry is a
    multipolygon, where each part of the multipolygon is a polygon. We want
    to select the biggest polygon, which is the real WS, and not a
    malfunctioned part of it.

    Parameters
    ----------
    ws_geom : shapely.geometry.MultiPolygon
        The multipolygon for which to select the biggest polygon.

    Returns
    -------
    shapely.geometry.Polygon
        The biggest polygon of the multipolygon.

    """
    if isinstance(ws_geom, MultiPolygon):
        # Compute the area of each polygon in the multipolygon
        areas = [polygon_area(geo_shape=polygon) for polygon in ws_geom.geoms]
        # Find the index of the polygon with the biggest area
        idx = np.argmax(areas)
        # Select the polygon with the biggest area
        ws_geom = ws_geom.geoms[idx]
    else:
        # If the geometry is not a multipolygon, just return it
        ws_geom = ws_geom
    return ws_geom


def ws_AOI(ws, shp_path: str):
    """Calculate the Area of Interest (AOI) from the given polygon and save it as a shapefile.

    Args:
    ----
    ws : Polygon
        The input polygon to determine the AOI from.
    shp_path : str
        The file path where the shapefile will be saved.

    Returns:
    -------
    str
        The path to the saved shapefile.

    """

    def my_ceil(a, precision=0):
        """Round up a number to the specified precision."""
        return np.true_divide(np.ceil(a * 10**precision), 10**precision)

    def my_floor(a, precision=0):
        """Round down a number to the specified precision."""
        return np.true_divide(np.floor(a * 10**precision), 10**precision)

    # Extract longitude and latitude from the polygon's exterior
    LONS, LATS = ws.exterior.xy

    # Calculate the maximum and minimum latitude and longitude
    max_LAT = np.max(LATS)
    max_LON = np.max(LONS)
    min_LAT = np.min(LATS)
    min_LON = np.min(LONS)

    # Adjust the boundaries with a specified precision
    min_LON, max_LON, min_LAT, max_LAT = (
        my_floor(min_LON, 2),
        my_ceil(max_LON, 2),
        my_floor(min_LAT, 2),
        my_ceil(max_LAT, 2),
    )

    # Create a GeoDataFrame for the AOI geometry
    test = gpd.GeoDataFrame()
    aoi_geom = Polygon([[min_LON, min_LAT], [min_LON, max_LAT], [max_LON, max_LAT], [max_LON, min_LAT]])
    test.loc[0, "geometry"] = aoi_geom  # type: ignore
    test = test.set_crs(epsg=4326)  # Set coordinate reference system to WGS 84

    # Save the AOI as a shapefile
    test.to_file(f"{shp_path}", index=False)

    return shp_path


def round_up(x, round_val: float = 5):
    """Round up a number to the nearest multiple of `round_val`.

    Args:
    ----
        x (float): The number to round up.
        round_val (float): The value to round up to. Defaults to 5.

    Returns:
    -------
        float: The rounded up number.

    """
    return int(np.ceil(x / round_val)) * round_val


def round_down(x, round_val: float = 5):
    """Round down a number to the nearest multiple of `round_val`.

    Args:
    ----
        x (float): The number to round down.
        round_val (float): The value to round down to. Defaults to 5.

    Returns:
    -------
        float: The rounded down number.

    """
    # Divide `x` by `round_val`, take the floor to round down, then multiply back by `round_val`
    return int(np.floor(x / round_val)) * round_val


def find_extent(ws: Polygon, grid_res: float, dataset: str = "") -> list:
    """Find the extent of the given polygon, rounded to the nearest multiple of the grid resolution.

    Args:
    ----
        ws (Polygon): The polygon for which to find the extent.
        grid_res (float): The grid resolution to round to.
        dataset (str, optional): The dataset to consider. Defaults to "".

    Returns:
    -------
        list: The extent of the polygon, rounded to the nearest multiple of the grid resolution.

    """

    def x_round(x):
        """Round to the nearest 0.25."""
        return round((x - 0.25) * 2) / 2 + 0.25

    def round_nearest(x, a):
        """Round to the nearest multiple of a."""
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
        # Round to the nearest 0.25 for GPCP
        return [x_round(min_LON), x_round(max_LON), x_round(min_LAT), x_round(max_LAT)]
    elif bool(dataset):
        # Round to the nearest multiple of the grid resolution for other datasets
        return [
            round_nearest(min_LON, grid_res),
            round_nearest(max_LON, grid_res),
            round_nearest(min_LAT, grid_res),
            round_nearest(max_LAT, grid_res),
        ]
    else:
        raise Exception(f"Something wrong ! {dataset} -- {grid_res}")


def create_gdf(shape: Union[Polygon, MultiPolygon]) -> DataFrame | gpd.GeoDataFrame:
    """Create GeoDataFrame from given shape.

    Args:
    ----
        shape (Union[Polygon, MultiPolygon]): The shape to create a GeoDataFrame from.

    Returns:
    -------
        gpd.GeoDataFrame: The created GeoDataFrame.

    """
    # Convert shape to a MultiPolygon
    gdf_your_WS = poly_from_multipoly(ws_geom=shape)

    # Create extra gdf to use geopandas functions
    gdf_your_WS = gpd.GeoDataFrame({"geometry": [gdf_your_WS]})

    # Set the crs to EPSG:4326
    gdf_your_WS = gdf_your_WS.set_crs("EPSG:4326")

    return gdf_your_WS


def RotM(alpha):
    """Rotation Matrix for angle ``alpha``.

    Args:
    ----
        alpha (float): The angle of rotation in radians.

    Returns:
    -------
        ndarray: The 2x2 rotation matrix.

    """
    sa, ca = np.sin(alpha), np.cos(alpha)
    return np.array([[ca, -sa], [sa, ca]])


def get_square_vertices(mm, h, phi):
    """Calculate the for vertices for square with center ``mm``, side length ``h`` and rotation ``phi``.

    Args:
    ----
        mm (ndarray): The center of the square in the format [longitude, latitude].
        h (float): The side length of the square.
        phi (float): The angle of rotation of the square in radians.

    Returns:
    -------
        ndarray: The vertices of the square in the format [[longitude, latitude], ...].

    """
    hh0 = np.ones(2) * h  # initial corner
    # rotate initial corner four times by 90°
    vv = [np.asarray(mm) + reduce(np.dot, [RotM(phi), RotM(np.pi / 2 * c), hh0]) for c in range(4)]
    return np.asarray(vv)


@jit(nopython=True)
def point_distance(new_lon, new_lat, old_lon, old_lat):
    """Calculate the distance between two points on Earth surface.

    Args:
    ----
        new_lon (float): The longitude of the point in radians.
        new_lat (float): The latitude of the point in radians.
        old_lon (float): The longitude of the reference point in radians.
        old_lat (float): The latitude of the reference point in radians.

    Returns:
    -------
        float: The distance between the two points in kilometers.

    """
    # Calculate the longitude and latitude differences
    dlon = old_lon - new_lon
    dlat = old_lat - new_lat

    # Calculate the Haversine distance
    a = sin(dlat / 2) ** 2 + cos(old_lat) * cos(new_lat) * sin(dlon / 2) ** 2
    c = 2 * arctan2(sqrt(a), sqrt(1 - a))

    # Return the distance in kilometers
    return 6373.0 * c


def inside_mask(pt: List[float], bnd: List[float]) -> bool:
    """Check if a point is inside a given bounding box.

    Args:
    ----
        pt (List[float]): The point to check in the format [longitude, latitude].
        bnd (List[float]): The bounding box in the format [lon_min, lat_min, lon_max, lat_max].

    Returns:
    -------
        bool: True if the point is inside the bounding box, False otherwise.

    """
    x1, y1, x2, y2 = bnd
    x, y = pt
    if x1 < x and x < x2:
        if y1 < y and y < y2:
            return True
    return False


def str_to_np(x):
    """Convert a string representation of a numpy array to a numpy array.

    This function takes a string representation of a numpy array, such as
    a string created with the numpy.array2string() function, and converts
    it to a numpy array.

    Args:
        x (str): The string representation of the numpy array.

    Returns:
        ndarray: The numpy array represented by the string.

    """
    # Replace all spaces with commas
    x = x.replace(" ", ",")

    # Convert the string to a numpy array using the ast.literal_eval() function
    return np.array(ast.literal_eval(x))


def update_geometry(pnt, tile):
    """Update the geometry of the point in tile.

    This function takes a point object and a tile path as input, and returns a
    GeoDataFrame containing the updated geometry of the point in the tile.

    Parameters
    ----------
    pnt : object
        The point object containing x (longitude) and y (latitude) attributes.
    tile : str
        The path to the tile file.

    Returns
    -------
    GeoDataFrame
        A GeoDataFrame containing the updated geometry of the point in the tile.

    """
    # Get the scale for each type of river
    scale_actual = {
        "small_rivers": 80,
        "medium_rivers": 800,
        "rivers": 8000,
        "big_rivers": 80000,
        "large_rivers": 800000,
    }

    # Read the tile file and get all the points in the tile
    point_in_tile = gpd.read_file(tile)

    # Get the longitude and latitude of the point
    old_lon = pnt.x
    old_lat = pnt.y

    # Convert the geometry of each point to a numpy array
    point_in_tile["geom_np"] = point_in_tile["geom_np"].apply(lambda x: str_to_np(x))

    # Calculate the distance of each point from the given point
    point_in_tile["distance"] = point_in_tile["geom_np"].apply(
        lambda x: point_distance(
            new_lon=radians(x[0]),
            new_lat=radians(x[1]),
            old_lon=radians(old_lon),
            old_lat=radians(old_lat),
        )
    )

    # Get the approximate area of each point based on the rank
    point_in_tile["approx_area"] = point_in_tile["rank"].apply(lambda x: scale_actual[x])

    # Sort the points by distance and approximate area, and return the first point
    res_tile = (
        point_in_tile.sort_values(by=["distance", "approx_area"], axis=0)
        .reset_index(drop=True)
        .loc[[0], ["geometry", "distance", "approx_area"]]
    )
    return res_tile
