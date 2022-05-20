# wheel-Rut-semantic-segmentation

This repo includes the scripts to replicate the methods developed in [Bhatnagar et al. (2022)](https://zenodo.org/record/5746878#.YoeAzKhBxaQ) to perform a semantic segmentation of wheel-ruts caused by forestry machinery based on drone RGB imagery 📷. 

![Picture1](https://user-images.githubusercontent.com/5663984/169524083-197f2a17-fbc9-4b87-b0fb-324217caade5.png)
Figure 1. Example of the input and output of the developed method.

# Installation

```
# create new environment
conda create -n wheel_ruts_segment python=3.7.12

# clone repo
git clone https://github.com/SmartForest-no/wheel-Rut-semantic-segmentation

# install requirements
cd wheel-Rut-semantic-segmentation
pip install -r requirements.txt
```

# Usage
To run the segmentation on a new drone orthomosaic  
