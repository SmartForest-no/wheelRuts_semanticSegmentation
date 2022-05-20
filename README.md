# wheel-Rut-semantic-segmentation

This repo includes the scripts to replicate the methods developed in [Bhatnagar et al. (2022)](https://zenodo.org/record/5746878#.YoeAzKhBxaQ) to perform a semantic segmentation of wheel-ruts caused by forestry machinery based on drone RGB imagery 📷. 

![Picture1](https://user-images.githubusercontent.com/5663984/169524083-197f2a17-fbc9-4b87-b0fb-324217caade5.png)
Figure 1. Example of the input and output of the developed method.

# Installation

```
# create new environment
conda create -n wheel_ruts_segment python=3.7
conda create env -f 

# activate the created environment
conda activate wheel_ruts_segment

# clone repo
git clone https://github.com/SmartForest-no/wheelRuts_semanticSegmentation

# install requirements
cd wheelRuts_semanticSegmentation
pip install -r requirements.txt
```
In addition:
- Download model weights and json file at: https://drive.google.com/drive/folders/1byb7jcAPiB9pr2gunJCbec7fRiwzI6BG?usp=sharing
- Unzip the two files and and place them in the model folder

# Usage
## Input
The algorithm takes as input one orthomosaic or a folder with several orthomosaics and

## Output

## How to run
To run the segmentation on a new drone orthomosaic  
