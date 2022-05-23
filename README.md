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
The algorithm takes as input one orthomosaic or a folder with several orthomosaics in GeoTiff format (.tif)

## Output 🚜
The output consist of a binary raster with the same extent as the input orthomosaic where pixels with value of 1 correspond to wheel ruts and of value 0 correspond to background.

## How to run 🏃
To run the segmentation on a new drone orthomosaic run:
```
python run.py
```
This will open a window where you can select one orthomosaic to predict on. The default version (file_mode) allows you to select a single file but if you want to switch to the mode where is possible to feed an entire directory where several othomsaics are stored, then you should edit the ```run.py``` file by replacing ```file_mode``` with ```directory_mode```.

# Additional information
## Basic model
The base model has been trained for 50 epoch on the entire dataset described by [Bhatnagar et al. (2022)](https://zenodo.org/record/5746878#.YoeAzKhBxaQ). 
we should fill in with some accuracy figures.......

| model_name  | description | accuracy |
| ------------- | ------------- | ------------- |
| singleTrack_allData_25epochs  | output segmentation is a single track (model trained for 25 epochs) | 9999 |
| singleTrack_allData_49epochs  | output segmentation is a single track (model trained for 25 epochs) | 9999 |
| doubleTrack_32epochs  | output segmentation is a double track (model trained for 32 epochs) | 9999 |


## Training with your data
Saheba can you please fill in here ...........
