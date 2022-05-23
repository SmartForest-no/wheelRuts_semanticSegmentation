from keras_segmentation.models.unet import resnet50_unet #or any other model (see comments at the end of the doc)

#number of classes #no. of class is 'no. of labels + 1' as NaN values was given label ==0;
n = 3

model = resnet50_unet(n_classes=n ,  input_height=1024, input_width=1024  ) #input height, width is adjustable based on the area under consideration
#recommendation: 1024 x 1024 

epochs = 50

model.train(
    train_images =  "wheelRuts_semanticSegmentation/data/train/image/",
    train_annotations = "wheelRuts_semanticSegmentation/data/train/label/",
    validation_images =  "wheelRuts_semanticSegmentation/data/validation/image/", #optional
    validation_annotations = "wheelRuts_semanticSegmentation/data/validation/label/", #optional
    checkpoints_path = "weights/singleTrack_allData_25epochs" , #give a new name if training from scratch / use the name from the folder wheelRuts_semanticSegmentation/model to build on pre-trained model.
    epochs=epochs)


### USING DIFFERENT MODELS - OTHER THAN RESNET50+UNET ## 
## ORIGINAL REPOSITORY FOR CNN MODELS-- https://github.com/divamgupta/image-segmentation-keras ##
#model_name	
#fcn_8	             
#fcn_32	           
#fcn_8_vgg	          
#fcn_32_vgg	  	    
#fcn_8_resnet50	
#fcn_32_resnet50	
#fcn_8_mobilenet	
#fcn_32_mobilenet	
#pspnet	
#vgg_pspnet	
#resnet50_pspnet	
#unet_mini	
#unet	
#vgg_unet	
#resnet50_unet	
#mobilenet_unet
#segnet
#vgg_segnet
#resnet50_segnet
#mobilenet_segnet
