from typing import Union
from osgeo import gdal
from pathlib import Path


def reproject_and_clip(
    input_raster, output_raster, projection, shapefile: str = "", resolution: float = 0.0
):
    if resolution:
        if shapefile:
            options = gdal.WarpOptions(
                cutlineDSName=shapefile,
                cropToCutline=True,
                format="GTIFF",
                dstSRS=projection,
                xRes=resolution,
                yRes=resolution,
            )
        else:
            options = gdal.WarpOptions(
                cropToCutline=True, format="GTIFF", dstSRS=projection, xRes=resolution, yRes=resolution
            )
    else:
        if shapefile:
            options = gdal.WarpOptions(
                cutlineDSName=shapefile, cropToCutline=True, format="GTIFF", dstSRS=projection
            )
        else:
            options = gdal.WarpOptions(cropToCutline=True, format="GTIFF", dstSRS=projection)

    gdal.Warp(srcDSOrSrcDSTab=input_raster, destNameOrDestDS=output_raster, options=options)

    return output_raster


def create_mosaic(file_path: Union[Path, str], file_name: str, tiles: list) -> str:
    """Generate vrt mosaic for GDAL procedure from each .tiff for extent.

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
    mosaic = gdal.BuildVRT(destName=file_target, srcDSOrSrcDSTab=tiles)
    mosaic.FlushCache()

    return file_target


def vrt_to_geotiff(vrt_path: str, geotiff_path: str):
    src_ds = gdal.Open(vrt_path, 0)  # open the VRT in read-only mode
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
