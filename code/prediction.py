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
from keras.layers import Conv2D, UpSampling2D, Input, Flatten, Reshape, concatenate
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.layers.core import RepeatVector
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

# Global variables
path = "./lfw" # Total number of images is 13233 (currently only covering first 12288 images)
file_ending = ".jpg"
original_image_size = 250
image_size = 256
max_training_images = 12000
max_testing_images = 16
batch_size = 4 # Batch size should divide max_training_images and max_testing_images without remainders (for nice data)

# provides side information for every 8x8 block (Cb and Cr value)
def getSideInformation(image, orig):
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

# Get the original image and the predicted image and print them out
def testing_print(prediction, start, end):
    images = []
    num_images = 0
    file_names = []
    for file_name in glob.glob(path + "/*/*" + file_ending):
        if num_images == end:
            break
        elif num_images >= start:
            original_image = Image.open(file_name).convert("YCbCr", dither = None) # Add dithering later? (idea from Amar)
            image = Image.new("YCbCr", (image_size, image_size), (0, 128, 128))
            image.paste(original_image, (0, image_size - original_image_size)) # Extends image from 250x250 to 256x256 (shift original to start at bottom left, the rest is black)
            images.append(img_to_array(image))
            image.close() # Safety
            original_image.close() # Safety
            file_names.append(file_name)
            num_images = num_images + 1
        else:
            num_images = num_images + 1
    images = np.array(images, dtype = float)

    # Re-Normalize data 
    prediction = (prediction * 128.0) + 128.0

    # Output predictions(testing) and originals
    if not os.path.exists("results"):
        os.mkdir("results")
    
    for index in range(end - start):
        # Testing image
        testing_full_image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
        testing_full_image[..., 0] = images[index,:,:,0]
        testing_full_image[..., 1:] = prediction[index,...].reshape(image_size, image_size, 2)
        testing_full_image = Image.fromarray(np.uint8(testing_full_image), mode = "YCbCr")
        filename, file_extension = os.path.splitext(os.path.basename(file_names[index]))
        testing_full_image.convert("RGB").save("./results/predicted_image_"+ filename + ".png")
        # Original image
        original_image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
        original_image = images[index,...]
        original_image = Image.fromarray(np.uint8(original_image), mode = "YCbCr")
        original_image.convert("RGB").save("./results/original_image_"+ filename + ".png")

# Get Testing Data
# Changes image to 256 x256 since that is what CNN network expects
# training data input (x) is gotten from end-new_start number of images
# x[0] is the gray image
# x[1] is compressed Cb, x[2] is compress Cr
# x[3] is the edge image
new_start = max_training_images
end = max_training_images + max_testing_images
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
        side_information.append(getSideInformation(img_to_array(temp_side_image), img_to_array(image)))
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
x = np.zeros((max_testing_images, image_size, image_size, 4), dtype = float)
x[...,0] = images[...,0]
x[...,1:3] = side_information
x[...,3] = edges.reshape(max_testing_images, image_size, image_size)

# Check if old_model exists, if not create a new model. Otherwise train on the old model
if os.path.isfile("my_model.h5"):
    print("Loading existing model...")
    model = load_model("my_model.h5")
    model.load_weights("my_model.h5")
    y_prediction = model.predict(x=x, batch_size=batch_size)
    testing_print(y_prediction, max_training_images, max_training_images+max_testing_images)
else:
    print("No existing model found!")