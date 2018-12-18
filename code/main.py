# Data set used from http://vis-www.cs.umass.edu/lfw/ (All images as gzipped tar file) provides 13233 images of faces (more here: http://vis-www.cs.umass.edu/lfw/README.txt)
# Python version (windows): Python 3.6.7 - 2018-10-20 (download from https://www.python.org/downloads/release/python-367/ the "Windows x86-64 executable installer" version)
# Commands used:
# pip install Pillow
# pip install tensorflow (maybe can update to the following to get gpu working: pip install tensorflow-gpu)
# pip install keras

# Add cuDNN to allow keras to use GPU's??? (does that mean don't need tensorflow gpu support, or do I? https://docs.nvidia.com/deeplearning/sdk/cudnn-install/)
# In order to use keras with a GPU, the following was used used:
# Gefore GTX 745 GPU
# Latest NVIDIA Driver download
# CUDA TOOLKIT 9.0.176 (Windows, x86_64)
# cuDNN v7.3.1 for CUDA 9.0 - sept 28, 2018 (windows 10 version)
# SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;%PATH%
# SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64;%PATH%
# SET PATH=C:\Program Files\cuda\bin;%PATH%
# pip install tensorflow-gpu

# IMPORTANT~~~~~~~~~~~~~~~~~~~~~
# The scripts freeze_graph.exe, saved_model_cli.exe, tensorboard.exe, tflite_con
# vert.exe, toco.exe and toco_from_protos.exe are installed in 'c:\users\michael\a
# ppdata\local\programs\python\python36\Scripts' which is not on PATH.
# Consider adding this directory to PATH or, if you prefer to suppress this warn
# ing, use --no-warn-script-location.
# Therefore I did this:
# SET PATH=C:\users\michael\appdata\local\programs\python\python36\Scripts;%PATH%

# To use plot_model(...), do:
# pip install pydot
# and install graphviz from here:
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html (mine is 2.38 stable release)
# Add as path:
# SET PATH=C:\Program Files\graphviz\release\bin;%PATH%


from PIL import Image, ImageFilter
import glob
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, UpSampling2D, Input, Flatten, Reshape, concatenate, SeparableConv2D, MaxPooling2D, BatchNormalization, Activation, add
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.layers.core import RepeatVector
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

# Global variables
path = "./lfw/" # Total number of images is 13233
file_ending = ".jpg"
original_image_size = 250
image_size = 256
max_training_images = 2048 # 2048 produces better results (0.0015 MSE vs 0.0019)
max_validation_images = 4
batch_size = 4
num_epochs = 100

# provides side information for every 8x8 block (Cb and Cr value)
def getSideInformation(image):
    side_image = image
    side_image = Image.fromarray(np.uint8(side_image), mode = "YCbCr")
    side_image = side_image.resize((256, 256))
    side_image = side_image.convert("RGB")
    side_image = side_image.filter(ImageFilter.GaussianBlur(1)) # TODO Can change this parameter
    side_image = side_image.convert("YCbCr")
    side_info = img_to_array(side_image)
    side_image.close() # Safety
    side_info = side_info[...,1:]
    return side_info

# Get Training Data
# Changes image to 256 x256 since that is what CNN network expects
# training data input (x_train) is gotten from end-new_start number of images
# x_train[0] is the gray image
# x_train[1] is compressed Cb, x_train[2] is compress Cr
# x_train[3] is the edge image
# y_train is the expected output (original image)
# the same is done for the validation images in x_valid and y_valid
new_start = 0
end = max_training_images + max_validation_images + new_start
images = []
num_images = 0
side_information = [] # provides side information of the CbCr value (https://en.wikipedia.org/wiki/File:YCbCr-CbCr_Scaled_Y50.png). 0 = top left, 1 = top right, 2 = bot left, 3 = bot right
edges = []
for file_name in glob.glob(path + "/*/*" + file_ending):
    if num_images == end:
        break
    elif num_images >= new_start:
        original_image = Image.open(file_name).convert("YCbCr", dither = None) # Add dithering later? (idea from Amar)
        image = Image.new("YCbCr", (image_size, image_size), (0, 128, 128))
        image.paste(original_image, (0, image_size - original_image_size)) # Extends image from 250x250 to 256x256 (shift original to start at bottom left, the rest is black)
        temp_side_image = image.resize((round(image_size/32), round(image_size/32))) # For 256x256 image, compresses down to 8x8
        images.append(img_to_array(image))
        side_information.append(getSideInformation(img_to_array(temp_side_image)))
        edges.append(img_to_array(image.convert("RGB").filter(ImageFilter.FIND_EDGES).convert("1"))) # Already normalized since values are either 0 or 1
        image.close() # Safety
        original_image.close() # Safety
        temp_side_image.close() # Safety
        num_images = num_images + 1
    else:
        num_images = num_images + 1
images = np.array(images, dtype = float)
images = (images - 128.0) / 128.0; # Normalized data from -1 to 1
edges = np.array(edges, dtype = float) # Already either 0 or 1
side_information = np.array(side_information, dtype = float)
side_information = (side_information - 128.0) / 128.0 # Normalized data from -1 to 1
x_train = np.zeros((max_training_images, image_size, image_size, 4), dtype = float)
x_train[...,0] = images[0:max_training_images,:,:,0]
x_train[...,1:3] = side_information[0:max_training_images,...]
x_train[...,3] = edges[0:max_training_images,...].reshape(max_training_images, image_size, image_size)
y_train = images[0:max_training_images,:,:,1:].reshape(max_training_images, image_size, image_size, 2)
x_valid = np.zeros((max_validation_images, image_size, image_size, 4), dtype = float)
x_valid[...,0] = images[max_training_images:max_training_images+max_validation_images,:,:,0]
x_valid[...,1:3] = side_information[max_training_images:max_training_images+max_validation_images,...]
x_valid[...,3] = edges[max_training_images:max_training_images+max_validation_images,...].reshape(max_validation_images, image_size, image_size)
y_valid = images[max_training_images:max_training_images+max_validation_images,:,:,1:].reshape(max_validation_images, image_size, image_size, 2)

# Exception like model with backend being inversion of front end
model_input = Input(shape=(256, 256, 4))
model_output = Conv2D(32, (3, 3), strides=(2, 2),use_bias=False)(model_input)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = Conv2D(64, (3, 3), use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)

model_temp = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(model_output)
model_temp = BatchNormalization()(model_temp)

model_output = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model_output)
model_output = add([model_output, model_temp])

model_temp = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(model_output)
model_temp = BatchNormalization()(model_temp)

model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model_output)
model_output = add([model_output, model_temp])

model_temp = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(model_output)
model_temp = BatchNormalization()(model_temp)

model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(model_output)
model_output = add([model_output, model_temp])

for i in range(8):
    model_temp = model_output
    model_output = Activation('relu')(model_output)
    model_output = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(model_output)
    model_output = BatchNormalization()(model_output)
    model_output = Activation('relu')(model_output)
    model_output = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(model_output)
    model_output = BatchNormalization()(model_output)
    model_output = Activation('relu')(model_output)
    model_output = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(model_output)
    model_output = BatchNormalization()(model_output)
    model_output = add([model_output, model_temp])

# TODO TEST
model_temp = Conv2D(728, (1, 1), padding='same', use_bias=False)(model_output)
model_temp = BatchNormalization()(model_temp)
model_temp = UpSampling2D((2,2))(model_temp)

model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = UpSampling2D((2, 2))(model_output)
model_output = add([model_output, model_temp])

model_temp = Conv2D(256, (1, 1), padding='same', use_bias=False)(model_output)
model_temp = BatchNormalization()(model_temp)
model_temp = UpSampling2D((2,2))(model_temp)

model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = UpSampling2D((2, 2))(model_output)
model_output = add([model_output, model_temp])

model_temp = Conv2D(128, (1, 1), padding='same', use_bias=False)(model_output)
model_temp = BatchNormalization()(model_temp)
model_temp = UpSampling2D((2,2))(model_temp)

model_output = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = UpSampling2D((2, 2))(model_output)
model_output = add([model_output, model_temp])

model_output = Conv2D(64, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = Conv2D(32, (3, 3), padding='same', use_bias=False)(model_output)
model_output = BatchNormalization()(model_output)
model_output = Activation('relu')(model_output)
model_output = UpSampling2D((2, 2))(model_output)
model_output = Conv2D(2, (3, 3), activation='tanh', padding='same', use_bias=False)(model_output)

model = Model(model_input, model_output)
# print(model.summary())

# Check if old_model exists, if not create a new model. Otherwise train on the old model
if os.path.isfile("my_model.h5"):
    print("Loading existing weights...")
    model.load_weights("my_model.h5")

from keras.utils import plot_model
plot_model(model, show_shapes = True, to_file='model.png')

# Train model      
model.compile(optimizer='adam', loss='mse')

# Save whole models (includes: architecture + weights + optimizer state) in a checkpoint
checkpoint = ModelCheckpoint("my_model.h5")
callback = [checkpoint]

train_datagen = ImageDataGenerator()
train_datagen.fit(x_train)
valid_datagen = ImageDataGenerator()
valid_datagen.fit(x_valid)

model.fit_generator(train_datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=max_training_images/batch_size, epochs=num_epochs, callbacks=callback, 
    validation_data = valid_datagen.flow(x_valid, y_valid, batch_size=batch_size), validation_steps=max_validation_images/batch_size)
# model.fit(x=x, y=y, batch_size=batch_size, callbacks=callback, validation_split=0.2, epochs=num_epochs)
model.save("my_model.h5")
