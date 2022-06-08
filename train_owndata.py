## Training the wheel rut model for your own data ##

from keras_segmentation.models.unet import resnet50_unet #for running other pretrained architectures, please refer introduction of the repository
#number of classes #no. of class is 'no. of labels + 1' as NaN values was given label ==0;
n = 3

model = resnet50_unet(n_classes=n ,  input_height=1024, input_width=1024  ) #input height, width is adjustable based on the area under consideration
#recommendation: 1024 x 1024 

epochs = 50

#unzip data.zip to train the model with an example dataset

model.train(
    train_images =  "data/train/image/",
    train_annotations = "data/train/label/",
    val_images =  "data/validation/image/", #optional
    val_annotations = "data/validation/label/", #optional
    checkpoints_path = "model/singleTrack_allData_25epochs" , #give a new name if training from scratch / use the name from the folder wheelRuts_semanticSegmentation/model to build on pre-trained model.
    epochs=epochs)
