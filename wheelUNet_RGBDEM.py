from simple_multi_unet_model import orig_unet_model #Uses softmax 
#from PIL import Image
from libtiff import TIFF

import os
import glob
import cv2
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
#from tensorflow.python.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

#Resizing images, if needed
SIZE_X= 256
SIZE_Y= 256
n_classes=3 #Number of classes for segmentation

#Capture training image info as a list
train_images = []
for directory_path in glob.glob("/home/nibio/.../data_saheba/Wheelrut_VM/image_rgbdem/"):
    list_of_files = sorted( filter( os.path.isfile,
                        glob.glob(directory_path + '*.tif') ) )
    for img_path in list_of_files:
        im = TIFF.open(img_path)
        img = im.read_image()
        #img = cv2.imread(img_path)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        img[img<0] = 0
        train_images.append(img)
       
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
train_masks = [] 
for directory_path in glob.glob("/home/nibio/.../data_saheba/Wheelrut_VM/label_rgbdem/"):
    list_of_files1 = sorted( filter( os.path.isfile,
                        glob.glob(directory_path + '*.tif') ) )
    for mask_path in list_of_files1:
        im1 = TIFF.open(mask_path)
        mask = im1.read_image()
        #mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  
        mask[mask>2] = 1
        train_masks.append(mask)
        
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)

###############################################
#Encode labels... but multi dim array so need to flatten, encode and reshape

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_encoded_original_shape = train_masks_reshaped.reshape(n, h, w)
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
np.unique(train_masks_encoded_original_shape)

#################################################
train_images1 =  cv2.normalize(train_images, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

#Create a subset of data for quick testing
#Picking at least 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.1, random_state = 0, shuffle = True)

print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))

test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

### to balance the weights (but works only for binary segmentation (??????))
#from sklearn.utils import class_weight
#class_weights = class_weight.compute_class_weight('balanced',
#                                                 np.unique(train_masks_encoded_original_shape),
#                                                 train_masks_encoded_original_shape)
#
#class_weights = dict(zip(np.unique(train_masks_encoded_original_shape), class_weight.compute_class_weight('balanced', np.unique(train_masks_encoded_original_shape), 
#                train_masks_encoded_original_shape))) 
#
#print("Class weights are...:", class_weights)

## Defining model ##
IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return orig_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer= opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

#If starting with pre-trained weights. 
#model.load_weights('pretrained.hdf5')

checkpointer = tf.keras.callbacks.ModelCheckpoint('/pretrained.hdf5', verbose = 1, save_best_only = True)

callbacks = [tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_accuracy'), 
             tf.keras.callbacks.TensorBoard(log_dir='logs'),
             checkpointer]

history = model.fit(X_train, y_train_cat, 
                    batch_size = 25, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test_cat), 
                    callbacks = callbacks,
                    #class_weight=class_weights,
                    shuffle=True)

model.save('/newmodel.hdf5')

############################################################
#Evaluate the model
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

###
#plot the training and validation accuracy and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


##################################
model = get_model()

#IOU
y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)

##################################################

#Using built in keras function
from keras.metrics import MeanIoU
n_classes =3
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[1,0]+ values[2,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[0,1]+ values[2,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1]  + values[0,2]+ values[1,2])
#class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
#class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2] + values[0,4]+ values[1,4]+ values[2,4])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
#print("IoU for class4 is: ", class4_IoU)
#print("IoU for class5 is: ", class5_IoU)


#######################################################################
#Predict on a few validation images

import random
test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
#test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]

colors =  ['1','0','0.8']

plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,1])#, cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap=matplotlib.colors.ListedColormap(colors))
plt.subplot(233)
plt.title('Prediction on test image')
colors1 =  ['1','0','0.8']
plt.imshow(predicted_img, cmap=matplotlib.colors.ListedColormap(colors))
plt.show()

#####################################################################
## Predict on multiple test images ##

testa_images = []
for directory_path in glob.glob("/home/nibio/.../data_saheba/Wheelrut_VM/image_test_rgbdem/"):
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
for directory_path in glob.glob("/home/nibio/.../data_saheba/Wheelrut_VM/label_test_rgbdem/"):
    list_of_files2 = sorted( filter( os.path.isfile,
                        glob.glob(directory_path + '*.tif') ) )
    for mask_path in list_of_files2:
        im1 = TIFF.open(mask_path)
        mask1 = im1.read_image()
        #mask1 = cv2.imread(mask_path, 0)       
        mask1 = cv2.resize(mask1, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  
        mask1[mask1>2] = 1
        mask1[mask1<0] = 0
        testa_masks.append(mask1)
        
#Convert list to array for machine learning processing          
testa_masks = np.array(testa_masks)
n, h, w = testa_masks.shape
testa_masks_reshaped = testa_masks.reshape(-1,1)
testa_masks_reshaped = np.floor(testa_masks_reshaped)
testa_masks_encoded_original_shape = testa_masks_reshaped.reshape(n, h, w)

np.unique(train_masks_encoded_original_shape)
testa_masks_input = np.expand_dims(testa_masks_encoded_original_shape, axis=3)

from sklearn.metrics import f1_score
#from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix

listf1 = []
listcm = []

orininal_w = 1600 #original size of the tiles (20m)
orininal_h = 1600

image_no = 1

for i in range (0,len(list_of_files)):
    test_img_number = i
    test_img = testa_images[test_img_number]
    ground_truth=testa_masks_input[test_img_number]
    test_img_input=np.expand_dims(test_img, 0)
    prediction = (model.predict(test_img_input))
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
    name = '/home/nibio/.../data_saheba/Wheelrut_VM/test_origsize_image_'  + str(image_no) + '.png'
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

###############################################################################
## Predict on large image ##

# from patchify import patchify, unpatchify

# imbig = TIFF.open('.../biimage.tif')
# large_image = imbig.read_image()
# #This will split the image into small images of shape [3,3]
# patches = patchify(large_image, (128, 128, 10), step=128)  #Step=256 for 256 patches & no overlap

# predicted_patches = []
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         print(i,j)
        
#         single_patch = patches[i,j,:,:]       
#         #single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1),2)
#         #single_patch_input=np.expand_dims(single_patch_norm, 0)
#         single_patch_prediction = (model.predict(single_patch))#_input))
#         single_patch_predicted_img=np.argmax(single_patch_prediction, axis=3)[0,:,:]

#         predicted_patches.append(single_patch_predicted_img)

# predicted_patches = np.array(predicted_patches)

# predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 128,128) )

# reconstructed_image = unpatchify(predicted_patches_reshaped, (large_image.shape[0],large_image.shape[1]))
# plt.imshow(reconstructed_image, cmap='gray')

# plt.hist(reconstructed_image.flatten())  #Threshold everything above 0

# # 
# final_prediction = (reconstructed_image > 0.01).astype(np.uint8)
# plt.imshow(final_prediction)

# plt.figure(figsize=(8, 8))
# plt.subplot(221)
# plt.title('Large Image')
# plt.imshow(large_image[:,:,1])#, cmap='jet')
# plt.subplot(222)
# plt.title('Prediction of large Image')
# #plt.imshow(reconstructed_image, cmap='jet')
# #colors1 = ['0','1','0.8']
# plt.imshow(reconstructed_image, cmap=matplotlib.colors.ListedColormap(colors))
# plt.show()
