# Collection of tools/functions useful for object detection and semantic segmentation
import os, glob, shutil
from pathlib import Path
import numpy as np
#from cv2 import cv2
import cv2
import rasterio as rio
from rasterio.merge import merge
from rasterio.plot import show
from osgeo import gdal, ogr, osr
from keras_segmentation.predict import predict_multiple #importing predict function from keras 
import geopandas as gpd
import matplotlib.pyplot as plt
import tkinter.filedialog as fd
import tkinter as tk




def directory_mode():
    root = tk.Tk()
    ortho_to_process = []
    directory = fd.askdirectory(parent=root, title='Choose directory where orthomosaics (*.tif) are stored')
    ortho_to_process = glob.glob(directory + '/**/*.tif', recursive=True)
    root.destroy()
    return ortho_to_process


def file_mode():
    root = tk.Tk()
    ortho_to_process = fd.askopenfilenames(parent=root, title='Choose orthomosaic (*.tif) to process',
                                                  filetypes=[("TIF", "*.tif")])
    root.destroy()
    return ortho_to_process






##########################################################################################################################################################################
# function to split a large orthomosaic into small tiles for deep learning inference. The function allows to include a buffer around each tile so that the tiles are actually overlapping
# arguments: 
# - ortho_path    = path pointing at the orthomosaic geotiff file (e.g. "/home/datascience/cnn_wheel_ruts/data/my_ortho.tif")
# - tile_size_m   = length of the side of each tile in meters
# - buffer_size_m = width of the buffer around each tile. This is basically the area of overalp between each tile and is useful to avoid having excessive edge effects when running deep learning inference on small tiles.
# - format_tiles  = export format for tiles (either PNG or GTiff)
##########################################################################################################################################################################
def tile_ortho(ortho_path, tile_size_m, buffer_size_m, format_tiles):
       
    # 1 - DEFINE RASTER AND TILING PARAMETERS
    ## get name of the orthomosaic/drone project and the path where it's stored
    ortho_name=Path(ortho_path).stem # ortho name
    ortho_folder_path=os.path.dirname(ortho_path) # get path name for the folder where the orthomosaic is stored
    ## Get pixel resolution (in meters) and tile size in pixels
    src_ds = gdal.Open(ortho_path) # get raster datasource
    _, xres, _, _, _, yres  = src_ds.GetGeoTransform() # get pixel size in meters
    tile_size_px= round(tile_size_m/abs(xres)) # calculate the tile size in pixels
    ## Get EPSG code
    proj = osr.SpatialReference(wkt=src_ds.GetProjection())
    EPSG_code= proj.GetAttrValue('AUTHORITY',1)
    
    # 2 - GENERATE ORTHOMOSAIC BOUNDARY SHAPEFILE
    ## Define name for boundary shapefile
    shape_path=ortho_folder_path+"/"+ortho_name+"_boundary.shp"
    ## Run gdal_polygonize.py to get boundaries from alpha band (band 4)
    #%run /home/datascience/cnn_wheel_ruts/gdal_polygonize.py $ortho_path -b 4 $shape_path
    #os.chdir("/home/datascience/cnn_wheel_ruts/")

    os.chdir("scripts")
    command_polygonize = "gdal_polygonize.py "+ ortho_path + " -b 4 " + shape_path
    print(os.popen(command_polygonize).read())
    ## Select polygon that has DN equal to 255, indicating the area where drone data is available for
    polys = gpd.read_file(shape_path)
    polys[polys['DN']==255].to_file(shape_path)
    
    # 3 - TILING THE ORTHOMOSAIC
    ## Define buffer size and calculate the size of tiles excluding buffer
    buffer_size_m= 2 # size of buffer around each tile 
    tile_size_px= round(tile_size_m/abs(xres))
    buffer_size_px= round(buffer_size_m/abs(xres))
    ## Create folder for tiles to be exported in
    tiles_dir=ortho_folder_path+"/tiles_png"
    if not os.path.exists(tiles_dir): 
           os.makedirs(tiles_dir)    
    tileIndex_name=ortho_name+"_tile_index" # define name for output tile index shapefile
    ## Run gdal_retile.py (can take some minutes) 
    #os.chdir("/home/datascience/cnn_wheel_ruts/")
     #%run /home/datascience/cnn_wheel_ruts/gdal_retile.py -targetDir $tiles_dir $ortho_path -overlap $buffer_size_px -ps $tile_size_noBNuffer_px $tile_size_noBNuffer_px -of PNG -co WORLDFILE=YES -tileIndex $tileIndex_name -tileIndexField ID
    if format_tiles=="PNG":
        command_retile = "gdal_retile.py -targetDir " + tiles_dir + " " + ortho_path+ " -overlap " + str(buffer_size_px) + " -ps "+str(tile_size_px) + " " + str(tile_size_px) + " -of PNG -co WORLDFILE=YES -tileIndex "+ tileIndex_name + " -tileIndexField ID"
    if format_tiles=="GTiff":
        command_retile = "gdal_retile.py -targetDir " + tiles_dir + " " + ortho_path+ " -overlap " + str(buffer_size_px) + " -ps "+str(tile_size_px) + " " + str(tile_size_px) + " -of GTiff -tileIndex "+ tileIndex_name + " -tileIndexField ID"
    print(os.popen(command_retile).read())
    
   
    
    # 4 - KEEP ONLY TILES WITHIN THE ORTHOMOSAIC BOUNDARY
    ## Load boundary
    boundary = gpd.read_file(shape_path) #  read in the shapefile using geopandas
    boundary = boundary.geometry.unary_union #union of all geometries in the GeoSeries
    ## Load tiles shapefile
    tiles = gpd.read_file(tiles_dir+ "/"+ortho_name+"_tile_index.shp")
    ## Select all tiles that are not within the boundary polygon
    tiles_out = tiles[~tiles.geometry.within(boundary)]
    ## Create a series for each file format with all names of files to be removed
    names_tiles_out = [os.path.splitext(x)[0] for x in tiles_out['ID']] # get names without extension
    pngs_delete=[tiles_dir+ "/"+sub + '.png' for sub in names_tiles_out] # add .png extension
    xml_delete=[tiles_dir+ "/" +sub + '.png.tmp.aux.xml' for sub in names_tiles_out] # ...
    wld_delete=[tiles_dir+ "/"+sub + '.png.wld' for sub in names_tiles_out] #...
    ## Delete files
    for f in pngs_delete: # delete png files
        os.remove(f)
    for f in xml_delete:  # delete xmls files
        os.remove(f)
    for f in wld_delete:  # delete world files
        os.remove(f)


##########################################################################################################################################################################
# function to run inference for wheel rut semantic segmentation 
# arguments: 
# - png_dir= directory where png tiles (20 m side) are stored
##########################################################################################################################################################################
def predict_wheelRuts(png_dir, model_dir):
    
    # define output path
    output_dir= png_dir+'/predictions'
    
    # create folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    #os.chdir(model_dir)
    print("Model directory ................................"+model_dir+ "resnet_unet_rgb_20m_grp1_patch")
    # run inference
    predict_multiple( 
      checkpoints_path=model_dir + "resnet_unet_rgb_20m_grp1_patch" , #path to weights (stored model .json file)
      inp_dir=png_dir , #path to files to be predicted (.png images)
      out_dir=output_dir #path to predicted files - would be in .png format
    )
    
    #import matplotlib.pyplot as plt
    #plt.imshow(out)
    
    
##########################################################################################################################################################################
# function to mosaic raster predictions from semantic segmentation (merging 20 m sized tile prediction)
# arguments: 
# - predicted_dir  = directory where predictions tiles from predict_wheelRuts function (.png format) are stored 
# - dir_orig_tiles = directory where original images from gdal_retile.py  (.png format) are stored 
# - dir_export     = directory where to export the mosaic
##########################################################################################################################################################################
# function to mosaic predictions    
def mosaic_predictions_raster_semantic_seg(
    predicted_dir    # directory where predictions tiles from predict_wheelRuts function (.png format) are stored 
    , dir_orig_tiles # directory where original images from gdal_retile.py  (.png format) are stored 
    , dir_export
    , EPSG_code
    , ortho_name
):   # directory where to export the mosaic
   

    # 1 - PREPARE ENVIRONMENT FOR MOSAIC CREATION
    # move wlds from original images folder to prediction folder
    os.chdir(dir_orig_tiles)
    wlds=[]
    for file in glob.glob("*.wld"):
        predicted_dir_file=dir_orig_tiles+"/"+file
        wlds.append(predicted_dir_file)
    for j in wlds:    
        shutil.move(j, predicted_dir)
    # move xmls from original images folder to prediction folder
    xmls=[]
    for file in glob.glob("*.xml"):
        predicted_dir_file=dir_orig_tiles+"/"+file
        xmls.append(predicted_dir_file)
    for j in xmls:    
        shutil.move(j, predicted_dir)
    # get list of predicted tiles (*.png) 
    os.chdir(predicted_dir)
    pngs=[]
    for file in glob.glob("*.png"):
        pngs.append(file)
    # get list of world files (*.wld) related to the predicted tiles 
    wlds=[]
    for file in glob.glob("*.wld"):
        predicted_dir_file=predicted_dir+"/"+file
        wlds.append(predicted_dir_file)
    # get list of world files (*.wld) related to the predicted tiles 
    xmls=[]
    for file in glob.glob("*.xml"):
        predicted_dir_file=predicted_dir+"/"+file
        xmls.append(predicted_dir_file)
          
    # 2 - CONVERT PNG TILES TO GEOTIFF 
    os.chdir(predicted_dir) # change dir to prediction dir
    # iterate through each png tile and convert it to geotiff using rasterio 
    for i in pngs:
        # get metadata
        ## Load ESRI world file to extract metadata related to the geographical extent the tiles  
        wld_file= f = open(i+'.wld', 'r')
        wld_file=wld_file.read()
        XCellSize =float(wld_file.split()[0])
        YCellSize =float(wld_file.split()[3])
        WorldX=float(wld_file.split()[4])
        WorldY=float(wld_file.split()[5])
        ## Load png image to extract metadata related to the image size
        im = cv2.imread(i)
        Rows=im.shape[0]
        Cols=im.shape[1]
        ## get UTM coords of the upper left and low right corner of the png from the ESRI world file
        XMin = WorldX - (XCellSize / 2)
        YMax = WorldY - (YCellSize / 2) 
        XMax = (WorldX + (Cols * XCellSize)) - (XCellSize / 2)
        YMin = (WorldY + (Rows * YCellSize)) - (YCellSize / 2)

        # conversion
        ## gdal conversion (was working but now it does not trnasfer the coordinates correctly)
        #opt_translate=gdal.TranslateOptions(format="GTiff", bandList=([1]), projWin=[XMin, YMax, XMax,YMin], projWinSRS="EPSG:"+EPSG_code)
        #gdal.Translate(os.path.splitext(i)[0]+".tif", i, options= opt_translate)

        ## rasterio conversion
        dataset = rio.open(i) # open image
        bands = [1] # select only the first band
        data = dataset.read(bands) 
        ### create the output transform 
        west, south, east, north = (XMin, YMin, XMax, YMax)
        transform = rio.transform.from_bounds(west,south,east,north,
                                              data.shape[1],data.shape[2])
        ### set the output image kwargs
        kwargs = {
            "driver": "GTiff",
            "width": data.shape[1], 
            "height": data.shape[2],
            "count": len(bands), 
            "dtype": data.dtype, 
            "nodata": 0,
            "transform": transform, 
            "crs": "EPSG:"+EPSG_code
        }
        with rio.open(os.path.splitext(i)[0]+".tif", "w", **kwargs) as dst:
            dst.write(data, indexes=bands)
            
    # 3 - CREATE MOSAIC (USING EITHER GDAL OR RASTERIO)
    # get list of geotifs
    gtiffs=[]
    for file in glob.glob("*.tif"):
        gtiffs.append(predicted_dir+"/"+file)

    ## mosaic gtiffs using gdal.warp (rasterio seems to be working better!)
    #opt= gdal.WarpOptions(srcNodata=20, multithread=True, resampleAlg="max", srcSRS="EPSG:"+EPSG_code, dstSRS="EPSG:"+EPSG_code)
    #g= gdal.Warp(str(dir_export)+"/"+ortho_name+"mosaic.tif", gtiffs, format="GTiff", options=opt )
    #g = None # Close file and flush to disk
    
    ## mosaic gtiffs using gdal.warp (rasterio seems to be working better!)
    # define output file path and name
    out_fp = str(dir_export)+"/"+ortho_name+"mosaic.tif"
    # List for the source files
    src_files_to_mosaic = []
    # Iterate over raster files and add them to source -list in 'read mode'
    for fp in gtiffs:
        src = rio.open(fp)
        src_files_to_mosaic.append(src)
    # define custom function to merge rasters
    def custom_merge_works(old_data, new_data, old_nodata, new_nodata, index=None, roff=None, coff=None):
        old_data[:] = np.maximum(old_data, new_data)  # <== NOTE old_data[:] updates the old data array *in place*
    # Merge function returns a single mosaic array and the transformation info
    mosaic, out_trans = merge(src_files_to_mosaic, method=custom_merge_works) 
     # Copy the metadata
    out_meta = src.meta.copy()
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": mosaic.shape[1],
                     "width": mosaic.shape[2],
                     "transform": out_trans,
                     "crs": "EPSG:"+EPSG_code
                     }
                    )
    # Write the mosaic raster to disk
    with rio.open(out_fp, "w", **out_meta) as dest:
        dest.write(mosaic)


    # CLEANUP ENVIRONMENT
    # delete prediction folder
    os.chdir(predicted_dir)
    for j in pngs:    
        os.remove(j)   
    for j in wlds:    
        os.remove(j)
    for j in xmls:    
        os.remove(j) 
    for j in gtiffs:
        os.remove(j)
    shutil.rmtree(predicted_dir) 
    
    # delete dir_orig_tiles
    os.chdir(dir_orig_tiles)
    pngs=[]
    for file in glob.glob("*.png"):
        pngs.append(dir_orig_tiles+"/"+file)    
    for j in pngs:    
        os.remove(j) 
    shutil.rmtree(dir_orig_tiles) 
