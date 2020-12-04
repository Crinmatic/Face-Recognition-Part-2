from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
 
#resize all the images to this
IMAGE_SIZE = [224 , 224]
 
train_path = 'dataset\train'
valid_path = 'dataset\test'
 
#add preprocssing layer to the front of VGG
vgg = VGG16(input_shape = IMAGE_SIZE + [3] , weights = 'imagenet' , include_top = False)
#dont train existing weights
for layer in vgg.layers:
    layer.trainable = False
    
#useful of getting number of classes
folders = glob('dataset\train\*')
 
#our layers - more can be added if needed
x = Flatten()(vgg.output)
# x =  Dense(1000, activation ='relu')(x) 
prediction =  Dense(len(folders), activation = 'softmax')(x)
#create a model object model
model = Model(inputs = vgg.input, outputs = prediction)
model.summary()
#tell the model what cost and optimization method to use
model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])
 
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range= 0.2,
                                   horizontal_flip =True)
test_datagen =  ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
                                                                                                                                                                                                                                                                                                                     







test_set = test_datagen.flow_from_directory(valid_path,
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
#fit the model
r = model.fit(
    training_set,
    validation_data = test_set,
    epochs = 20,
    steps_per_epoch = len(training_set),
    validation_steps = len(test_set)
    )
from keras.models import load_model
model.save('model.h5')