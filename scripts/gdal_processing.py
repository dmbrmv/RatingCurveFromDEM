"""Functions for gdal processing of tiffs.

This module contains functions for creating mosaic, cropping tiffs, creating
GeoTIFF and converting VRT to GeoTIFF.

"""

from pathlib import Path
from typing import Union
import geopandas as gpd
from osgeo import gdal


def create_mosaic(file_path: Union[Path, str], file_name: str, tiles: list) -> str:
    """Generate vrt mosaic for GDAL procedure from each .tiff for extent.

    This function takes a list of tiles (paths to .tiff files) and generates a
    virtual raster mosaic (VRT) that can be used by GDAL to process the data.

    Args:
    ----
        file_path (Union[Path, str]): Path for results
        file_name (str): Name for file (without extension)
        tiles (list): List of .tiff files from which we'll crop result

    Returns:
    -------
        str: Path for created mosaic

    """
    file_target = f"{file_path}/{file_name}.vrt"
    # Build a virtual raster mosaic (VRT) from the list of tiles
    mosaic = gdal.BuildVRT(destName=file_target, srcDSOrSrcDSTab=tiles)
    # Flush the cache to ensure that the mosaic is written to disk
    mosaic.FlushCache()

    return file_target


def vrt_to_geotiff(vrt_path: str, geotiff_path: str) -> str:
    """Convert a VRT mosaic to a GeoTIFF file.

    Args:
    ----
        vrt_path (str): Path to the VRT mosaic
        geotiff_path (str): Path for the output GeoTIFF file

    Returns:
    -------
        str: Path to the output GeoTIFF file

    """
    # Open the VRT in read-only mode
    src_ds = gdal.Open(vrt_path, 0)
    # Use gdal.Translate to convert the VRT to a GeoTIFF
    gdal.Translate(
        geotiff_path,
        src_ds,
        format="GTiff",
        creationOptions=["COMPRESS:DEFLATE", "TILED:YES"],
        callback=gdal.TermProgress_nocb,
    )
    # Properly close the datasets to flush to disk
    src_ds = None
    return geotiff_path


def gdal_extent_clipper(
    initial_tiff: str, extent: tuple, tmp_tiff: str, final_tiff: str, crs_epsg: int
) -> None:
    """Clip and reproject .tiff file for desired extent and epsg.

    This function takes a .tiff file, clips it to the desired extent, and then
    reprojects it to the desired EPSG code.

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
    # Clip big tiff for extent
    merged_tiff = gdal.Translate(destName=tmp_tiff, srcDS=initial_tiff, projWin=extent)
    # Flush the cache to ensure that the clipped data is written to disk
    merged_tiff.FlushCache()
    # Reproject the clipped data to the desired EPSG code
    merged_tiff_proj = gdal.Warp(
        destNameOrDestDS=final_tiff,
        format="GTiff",
        dstNodata=None,
        srcDSOrSrcDSTab=tmp_tiff,
        dstSRS=f"EPSG:{crs_epsg}",
    )
    # Flush the cache to ensure that the reprojected data is written to disk
    merged_tiff_proj.FlushCache()

    return None


def clip_raster_by_vector(raster_file_path: str, vector_file_path: str, output_file_path: str) -> None:
    """Clip a raster by a vector polygon using GDAL.

    Args:
        raster_file_path (str): Input raster file (GeoTIFF)
        vector_file_path (str): Input vector file (GPKG)
        output_file_path (str): Output clipped raster file (GeoTIFF)

    Returns:
        None

    """
    # # Load the clipping geometry from the GPKG file
    # clipping_geometry = gpd.read_file(vector_file_path).geometry.union_all()

    # # Convert the geometry to WKT format for use with GDAL
    # clipping_wkt = clipping_geometry.wkt

    # Clip the raster using gdal.Warp with the cutline
    warp_options = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=vector_file_path,
        cropToCutline=True,
    )

    # Perform the clipping
    gdal.Warp(destNameOrDestDS=output_file_path, srcDSOrSrcDSTab=raster_file_path, options=warp_options)

    return None