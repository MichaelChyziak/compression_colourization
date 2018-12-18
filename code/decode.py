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
import math

# Global variables
image_size = 256


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

file_name = ""
for file in glob.glob("./encoded/*" + ".png"): #Should only have a single png file there
# for file in glob.glob("./encoded/*" + ".jpg"): #Use this if grey image is jpg
    file_name = file

# Greyscale image
grey_image = Image.open(file_name)

# Edge image
edge_image = grey_image.filter(ImageFilter.FIND_EDGES).convert("1")
edge = img_to_array(edge_image)

# Recover Side information
file = open("./encoded/encoded_internal_side_info.txt", "rb")
side_info = file.read()
file.close()
side_info = list(side_info)
small_image = np.zeros((8, 8, 3), dtype = int)
small_grey = img_to_array(grey_image.resize((8, 8)))
small_image[...,0] = small_grey.reshape(8, 8)
for index in range(len(side_info)):
    if index % 2 == 0:
        # print(small_image[...,1])
        small_image[math.floor(index / 16), math.floor((index / 2) % 8), 1] = side_info[index]
    else:
        small_image[math.floor((index - 1) / 16), math.floor(((index - 1) / 2) % 8), 2] = side_info[index]
side_image = Image.fromarray(np.uint8(small_image), mode = "YCbCr")
side_image = side_image.resize((256, 256))
side_image = side_image.convert("RGB")
side_image = side_image.filter(ImageFilter.GaussianBlur(1)) # TODO Can change this parameter
side_image = side_image.convert("YCbCr")
side_info = img_to_array(side_image)
side_image.close() # Safety
side_info = side_info[...,1:]

# Combine greyscale, edge, and side info to be put into network
images = np.array(grey_image, dtype = float)
images = (images - 128.0) / 128.0; # Normalized data from -1 to 1
edges = np.array(edge, dtype = float) # Already either 0 or 1
side_information = np.array(side_info, dtype = float)
side_information = (side_information - 128.0) / 128.0 # Normalized data from -1 to 1
x = np.zeros((1, image_size, image_size, 4), dtype = float)
x[...,0] = images
x[...,1:3] = side_information
x[...,3] = edge.reshape(1, image_size, image_size)

# Check if model exists. Predict the model based on the input x and save as final output
if os.path.isfile("my_model.h5"):
    print("Loading existing model...")
    model = load_model("my_model.h5")
    model.load_weights("my_model.h5")
    prediction = model.predict(x=x)

    # Re-Normalize data 
    prediction = (prediction * 128.0) + 128.0

    # Output predictions(testing) and originals
    if not os.path.exists("decoded"):
        os.mkdir("decoded")
    
    # Original image
    final_image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
    final_image[..., 0] = grey_image
    final_image[..., 1:] = prediction.reshape(image_size, image_size, 2)
    final_image = Image.fromarray(np.uint8(final_image), mode = "YCbCr").convert("RGB")
    final_image.save("./decoded/final_image.png")
else:
    print("No existing model found!")
    exit()

# Output files
if not os.path.exists("decoded_demo"):
    os.mkdir("decoded_demo")

# Print Original image, side_image (for visualization purposes)
edge_image.convert("RGB").save("./decoded_demo/edge_image.png")