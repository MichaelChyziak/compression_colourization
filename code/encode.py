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
import os
import sys
from keras.preprocessing.image import img_to_array

# Global variables
image_size = 256
original_image_size = 250

# Get input image original, compressed version (250x250), extended version (256x256) and the compressed side info (Cb, Cr) of the image
if len(sys.argv) != 2:
    print("Improper number of input arguments")
    exit()
file_name = sys.argv[1]
original_image = Image.open(file_name).convert("YCbCr")
width, height = original_image.size
compressed_image = original_image.resize((250, 250))
extended_image = Image.new("YCbCr", (image_size, image_size), (0, 128, 128))
extended_image.paste(original_image, (0, image_size - original_image_size)) # Extends image from 250x250 to 256x256 (shift original to start at bottom left, the rest is black)
side_image = extended_image.resize((round(image_size/32), round(image_size/32))) # For 256x256 image, compresses down to 8x8
side_info = img_to_array(side_image)[...,1:]
side_info = side_info.astype(int)
side_info = [item for sublist in side_info for item in sublist]
side_info = [item for sublist in side_info for item in sublist]
side_info = bytes(side_info)
side_image = side_image.resize((256, 256))
side_image = side_image.convert("RGB")
side_image = side_image.filter(ImageFilter.GaussianBlur(1))
side_image = side_image.convert("YCbCr")

# Output files
if not os.path.exists("encoded_demo"):
    os.mkdir("encoded_demo")
if not os.path.exists("encoded"):
    os.mkdir("encoded")

# Print Original image, side_image, etc. (for visualization purposes)
filename, file_extension = os.path.splitext(os.path.basename(file_name))
original_image.convert("RGB").save("./encoded_demo/original_image_"+ os.path.basename(file_name))
compressed_image.convert("RGB").save("./encoded_demo/compressed_image_"+ filename + ".png")
extended_image.convert("RGB").save("./encoded_demo/extended_image_"+ filename + ".png")
side_image.convert("RGB").save("./encoded_demo/side_image_"+ filename + ".png")

# Save greyscale image, side info in file
# extended_image.convert("L").save("./encoded/edge_image_"+ os.path.basename(file_name))
extended_image.convert("L").save("./encoded/grey_image.png")
# extended_image.convert("L").save("./encoded/grey_image.jpg")
file = open("./encoded/encoded_internal_side_info.txt", "wb")
file.write(side_info)