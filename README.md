# Drone based wheel-ruts semantic segmentation

This repo includes the scripts to replicate the methods developed in [Bhatnagar et al. (2022)](https://zenodo.org/record/5746878#.YoeAzKhBxaQ) to perform a semantic segmentation of wheel-ruts caused by forestry machinery based on drone RGB imagery 📷. 

![Picture1](https://user-images.githubusercontent.com/5663984/169524083-197f2a17-fbc9-4b87-b0fb-324217caade5.png)
Figure 1. Example of the input and output of the developed method.

# Installation

```
# clone repo
git clone https://github.com/SmartForest-no/wheelRuts_semanticSegmentation
cd wheelRuts_semanticSegmentation

# create new environment
conda env create -f environment_cnn_wheelRuts.yaml

# activate the created environment
conda activate wheelRuts

# install requirements
pip install -r requirements.txt
```
In addition:
- Download model weights and json file at: https://drive.google.com/drive/folders/1byb7jcAPiB9pr2gunJCbec7fRiwzI6BG?usp=sharing
- Unzip the two files and and place them in the model folder

# Usage 💻
## Input 🗺️ 
The algorithm takes as input one orthomosaic or a folder with several orthomosaics in GeoTiff format (.tif). The default version takes as input a single orthomosaic, to change see "How to run" section.

## Output 🚜
The output consist of a binary raster with the same extent as the input orthomosaic where pixels with value of 1 correspond to wheel ruts and of value 0 correspond to background.

## How to run 🏃
To run the segmentation on a new drone orthomosaic run:
```
python run.py
```
This will open a window where you can select one orthomosaic to predict on. The default version (file_mode) allows you to select a single file but if you want to switch to the mode where is possible to feed an entire directory where several othomsaics are stored, then you should edit the ```run.py``` file by replacing ```file_mode``` with ```directory_mode```.

### additional run options
#### Use different tile size
Select the tile size and buffer size to split the original orthomosaic into smaller tiles by using the arguments ```--tile_size_m``` (default is 20 m) and ```--buffer_size_m``` (default is 2 m), e.g.:
```
python run.py --tile_size_m 20 --buffer_size_m 2

```
#### Select a different model
Select the model to run by using the argument ```--model_name``` (see next section for available models), e.g.:
```
python run.py --model_name doubleTrack_32epochs

```

# Available models
As 🍒 on top of the 🎂, in addition to the default model, we also provide additional models (see Table below). The default model (singleTrack_allData_49epochs) has been trained for 50 epoch on the entire dataset described by [Bhatnagar et al. (2022)](https://zenodo.org/record/5746878#.YoeAzKhBxaQ).  

| model_name  | description | accuracy |
| ------------- | ------------- | ------------- |
| singleTrack_allData_25epochs  | output segmentation is a single track (model trained for 25 epochs) | 9999 |
| singleTrack_allData_49epochs  | output segmentation is a single track (model trained for 25 epochs) | 9999 |
| doubleTrack_32epochs  | output segmentation is a double track (model trained for 32 epochs) | 9999 |


To select the different models edit in ```run.py``` the ```model_name``` variable to fit your preferred models. The doubleTrack model can be interesting for some as it produces a segmentation for each single track of the forestry machines (see image below).


# Training with your data
In case you want to re-train the model using your own data
```
python train_owndata.py
```
An example dataset has been attached (data.zip) for understanding the layout of the training and validation images.
Fully labelled images required for semantic segmentation (png/jpg)

For changing the training parameters, see wheelRuts_semanticSegmentation/keras_segmentation/train.py line 55 to make changes in batch size, optimizer, augmentation, etc.

# Using different architecture:

Original repository for CNN models (keras_segmentation)-- https://github.com/divamgupta/image-segmentation-keras 

```
In wheelRuts_semanticSegmentation/train_owndata.py line 3, change:
from keras_segmentation.models.'decoder' import 'encoder'
```
replace the decoder with:
fcn | unet | segnet | vgg16 | mobilenet | pspnet

replace the encoder name with:
fcn_8 | fcn_32 | fcn_8_vgg | fcn_32_vgg | fcn_8_resnet50 | fcn_32_resnet50 |
fcn_8_mobilenet| fcn_32_mobilenet | pspnet | vgg_pspnet | resnet50_pspnet | unet_mini |
unet | vgg_unet | resnet50_unet | mobilenet_unet | segnet | vgg_segnet |
resnet50_segnet | mobilenet_segnet |

# Going beyond 3 bands (RGB) 
For using multispectral images (for example, RGB + DEM, i.e., 4 bands), tensorflow based UNET architecture (wheelRuts_semanticSegmentation/orig_unet_model.py). The UNET architecture is based on the original UNET model (https://arxiv.org/abs/1505.04597). Can handle any dimension of data (multispectral) with variable sizing. All the tiles are resampled to a specific pixel size (height x width) by default in pre-processing (user defined).

The previously installed environment should be sufficient to run the script, but in case there are error, install new requirements. 
```
#install new requirements (if the previous environment doesn't work)
pip install -r requirements_tf.txt
```
Download the sample RGB+DEM 4 bands images + labels from data_rgbdem.zip 

To create, train and test the model:
```
python wheelUNet_RGBDEM.py 
```
To test directly on the test data using a pre-trained model:
- Download model (.hdf5) file from: https://drive.google.com/drive/folders/1byb7jcAPiB9pr2gunJCbec7fRiwzI6BG?usp=sharing
- Unzip the two files and and place them in the model folder

```
#to directly predict using pre-trained model for RGB + DEM (4 bands) imagery
python wheelUNet_RGBDEM_trainedmodel.py
```

Please make sure the paths of the images, labels, and weights is given properly in the script. 



