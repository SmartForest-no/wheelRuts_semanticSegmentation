from libtiff import TIFF
import os
import glob
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf

# .../ has to be replaced with the path of the directory in line 13, 23, 39, 96.

#loading the model
new_model = tf.keras.models.load_model('.../wheelmodelDEM_onlytracks.hdf5') #give path to the model directory
new_model.summary()

SIZE_X= 256
SIZE_Y= 256
n_classes=3

#preparing test images and labels

testa_images = []
for directory_path in glob.glob(".../image_test_rgbdem/"):
    list_of_files = sorted( filter( os.path.isfile,
                        glob.glob(directory_path + '*.tif') ) )
    for img_path in list_of_files:
        im = TIFF.open(img_path)
        img = im.read_image()
        #img = cv2.imread(img_path)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img[img<0] = 0
        testa_images.append(img)
       
#Convert list to array for machine learning processing        
testa_images = np.array(testa_images)
testa_images =  cv2.normalize(testa_images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

testa_masks = [] 
for directory_path in glob.glob(".../label_test_rgbdem/"):
    list_of_files2 = sorted( filter( os.path.isfile,
                        glob.glob(directory_path + '*.tif') ) )
    for mask_path in list_of_files2:
        im1 = TIFF.open(mask_path)
        mask1 = im1.read_image()
        #mask1 = cv2.imread(mask_path, 0)       
        mask1 = cv2.resize(mask1, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        mask1[mask1>2] = 1
        mask1[mask1<0] = 0
        #mask1 = int(mask1)
        testa_masks.append(mask1)
        
#Convert list to array for machine learning processing          
testa_masks = np.array(testa_masks)
n, h, w = testa_masks.shape
testa_masks_reshaped = testa_masks.reshape(-1,1)
testa_masks_reshaped = np.floor(testa_masks_reshaped)
testa_masks_encoded_original_shape = testa_masks_reshaped.reshape(n, h, w)
testa_masks_input = np.expand_dims(testa_masks_encoded_original_shape, axis=3)

from sklearn.metrics import f1_score
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

listf1 = []
listcm = []

orininal_w = 1600 #enter original width and height of the tiles
orininal_h = 1600

#prediction and measuring accuracy per image 
#saving resized images

image_no = 1
#import random
for i in range (1,245):
    test_img = testa_images[i]
    ground_truth=testa_masks_input[i]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (new_model.predict(test_img_input))
    predicted_img=np.argmax(prediction, axis=3)[0,:,:]

    colors = ['1','0','0.8']

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test_img[:,:,1])#, cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,0], cmap=matplotlib.colors.ListedColormap(colors))
    plt.subplot(233)
    plt.title('Prediction on test image')
    colors1 = ['1','0','0.8']
    plt.imshow(predicted_img, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()
    name = '.../predicted_test_origsize_image_'  + str(image_no) + '.png'
    predicted_img_origsize= cv2.resize(predicted_img, (orininal_w, orininal_h), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(name,predicted_img_origsize)
    image_no += 1
    gt1 = np.reshape(ground_truth, (256,256))
    gt2 = np.argmax(gt1, axis =1)
    pred2 = np.argmax(predicted_img, axis =1 )
    conftemp = f1_score(gt2,pred2, average='weighted')
    listf1.append(conftemp)
    gt3 = np.reshape(ground_truth, (256*256,1))
    pr3 = np.reshape(predicted_img, (256*256,1))
    confmat = multilabel_confusion_matrix(gt3,pr3)
    if np.size(np.unique(gt3))==3:
        confmat = confmat[2]
    else:
        confmat = confmat[1]
    listcm.append(confmat)
    
dem_test=listcm

tp = []
fp = []
fn = []
tn = []
for i in range(np.shape(listcm)[0]):
    t1 = dem_test[i][0,0]
    t2 = dem_test[i][0,1]
    t3 = dem_test[i][1,0]
    t4 = dem_test[i][1,1]
    tp.append(t1)
    fp.append(t2)
    fn.append(t3)
    tn.append(t4)

tpdem_test = sum(tp)

fpdem_test = sum(fp)

fndem_test = sum(fn)

tndem_test = sum(tn)
