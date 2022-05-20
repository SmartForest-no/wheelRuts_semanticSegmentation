import os, glob,shutil
#os.chdir("/home/datascience/cnn_wheel_ruts")
from keras_segmentation.predict import predict_multiple #importing predict function from keras 
from osgeo import gdal, ogr, osr
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
import numpy as np
import cv2
# load my functions
#os.chdir("/home/datascience/utils")
from scripts.tools import tile_ortho, predict_wheelRuts, mosaic_predictions_raster_semantic_seg, file_mode, directory_mode


if __name__ == '__main__':
    """Choose one of the following or modify as needed.
    Directory mode will find all .tif files within a directory and sub directories
	
    File mode will allow you to select multiple .tif files within a directory.
    
    Alternatively, you can just list the orthomosaic file paths.
    
    
    """
    # ortho_to_process = directory_mode()
    # ortho_to_process = ['full_path_to_your_point_cloud.las', 'full_path_to_your_second_point_cloud.las', etc.]
    ortho_to_process = file_mode()

    for ortho_path in ortho_to_process:
        
        # define some basic parameters
        tile_size_m= 20 # length of the side of each tile in meters (should NOT change this as this is the size that has been used in the training)
        buffer_size_m= 2 # size of buffer around each tile 
        
        
        
        # 3 - Split large orthomosaic into small tiles (20 meters side)
        print("Tiling orthomosaic..........")
        tile_ortho(ortho_path, tile_size_m, buffer_size_m, format_tiles="PNG")
        
        
        
        # 4 - Inference on tiled pngs
        print("Predicting wheel-ruts..........")

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        tiles_dir=os.path.dirname(ortho_path)+"/tiles_dir"
        predict_wheelRuts(tiles_dir)
        
        
        
        # 5 - Mosaic results
        print("Mosaicking results..........")

        ## Get orthomosaic name and EPGS code
        tiles_dir=os.path.dirname(ortho_path)+"/tiles_dir"
        ### get name of the orthomosaic/drone project and the path where it's stored
        ortho_name=Path(ortho_path).stem # ortho name
        ortho_folder_path=os.path.dirname(ortho_path) # get path name for the folder where the orthomosaic is stored
        ### Get pixel resolution (in meters) and tile size in pixels
        src_ds = gdal.Open(ortho_path) # get raster datasource
        _, xres, _, _, _, yres  = src_ds.GetGeoTransform() # get pixel size in meters
        tile_size_px= round(tile_size_m/abs(xres)) # calculate the tile size in pixels
        ### Get EPSG code
        proj = osr.SpatialReference(wkt=src_ds.GetProjection())
        EPSG_code= proj.GetAttrValue('AUTHORITY',1)
        
        ## Define function parameters
        predicted_dir=tiles_dir+'/predictions'
        dir_orig_tiles=os.path.split(predicted_dir)[0]
        dir_export=os.path.split(dir_orig_tiles)[0]
        
        ## run mosaicking function
        mosaic_predictions_raster_semantic_seg(predicted_dir , dir_orig_tiles, dir_export, EPSG_code, ortho_name)
        
        
        
        
        # END
        
        
        
        
