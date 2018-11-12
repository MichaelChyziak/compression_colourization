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


from PIL import Image
import glob
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Conv2D, UpSampling2D, Input, Flatten, Reshape, concatenate
from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.layers.core import RepeatVector
from keras.callbacks import TensorBoard
import numpy as np

# Global variables
path = "lfw/"
file_ending = ".jpg"
original_image_size = 250
image_size = 256
max_num_images = 12

# provides side information of the CbCr value (https://en.wikipedia.org/wiki/File:YCbCr-CbCr_Scaled_Y50.png). -1 = top left, -0.33 = top right, 0.33 = bot left, 1 = bot right (already normalized)
def getSideInformation(image):
    side_information = np.zeros((image_size, image_size), dtype = float)
    for row in range(image_size):
        for col in range(image_size):
            if image[row][col][1] >= 128 and image[row][col][2] >= 128:
                side_information[row][col] = -1.0/3.0
            elif image[row][col][1] < 128 and image[row][col][2] >= 128:
                side_information[row][col] = -3.0/3.0
            elif image[row][col][1] >= 128 and image[row][col][2] < 128:
                side_information[row][col] = 3.0/3.0
            else: # image[row][col][1] < 128 and image[row][col][2] < 128:
                side_information[row][col] = 1.0/3.0
    return side_information


# Get images YCbCr, images side information, and images file name
images = []
num_images = 0
file_names = []
side_information = [] # provides side information of the CbCr value (https://en.wikipedia.org/wiki/File:YCbCr-CbCr_Scaled_Y50.png). 0 = top left, 1 = top right, 2 = bot left, 3 = bot right
for file_name in glob.glob(path + "/*/*" + file_ending):
    original_image = Image.open(file_name).convert("YCbCr", dither = None) # Add dithering later? (idea from Amar)
    image = Image.new("YCbCr", (image_size, image_size), (0, 128, 128))
    image.paste(original_image, (0, image_size - original_image_size)) # Extends image from 250x250 to 256x256 (shift original to start at bottom left, the rest is black)
    images.append(img_to_array(image))
    image.close() # Safety
    original_image.close() # Safety
    file_names.append(file_name)
    side_information.append(getSideInformation(images[num_images]))
    num_images = num_images + 1
    if num_images == max_num_images:
        break
images = np.array(images, dtype = float)
images = (images - 128.0) / 128.0; # Normalized data from 0 to 1
side_information = np.array(side_information, dtype = float) # Already normalized
YCbCr_training = np.zeros((max_num_images, image_size, image_size, 4), dtype = float)
YCbCr_training[...,0:3] = images
YCbCr_training[...,3] = side_information

#Encoder
encoder_input = Input(shape=(256, 256, 1))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

# Side
# Change these to be fully connected (Dense) layers instead, it seems like this is better?
side_input = Input(shape=(256, 256, 1))
side_output = Conv2D(8, (3,3), activation='relu', padding='same', strides=2)(side_input)
side_output = Conv2D(16, (3,3), activation='relu', padding='same')(side_output)
side_output = Conv2D(16, (3,3), activation='relu', padding='same', strides=2)(side_output)
side_output = Conv2D(32, (3,3), activation='relu', padding='same')(side_output)
side_output = Conv2D(32, (3,3), activation='relu', padding='same', strides=2)(side_output)
side_output = Conv2D(64, (3,3), activation='relu', padding='same')(side_output)
side_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(side_output)
side_output = Conv2D(32, (3,3), activation='relu', padding='same')(side_output)
side_output = Conv2D(32, (3,3), activation='relu', padding='same', strides=2)(side_output)
side_output = Conv2D(16, (3,3), activation='relu', padding='same')(side_output)
side_output = Flatten()(side_output)
side_output = Reshape([1024])(side_output)

# Fusion
fusion_output = RepeatVector(32 * 32)(side_output) 
fusion_output = Reshape(([32, 32, 1024]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

#Decoder
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
model = Model(inputs=[encoder_input, side_input], outputs=decoder_output)
# print(model.summary())

from keras.utils import plot_model
plot_model(model, show_shapes = True, to_file='model.png')

# Train model      
tensorboard = TensorBoard(log_dir="./output")
model.compile(optimizer='adam', loss='mse')

# datagen = ImageDataGenerator(
#         shear_range=0.4,
#         zoom_range=0.4,
#         rotation_range=40,
#         horizontal_flip=True)

#Generate training data
#batch_size = 10

# def image_YCbCr_gen(batch_size):
#     for batch_data in datagen.flow(YCbCr_training, batch_size=batch_size):
#         YCbCr_batch_data = batch_data[...,0:3]
#         side_batch_data = batch_data[...,3]
#         Y_side_batch = np.zeros((batch_size, image_size, image_size, 2), dtype = float)
#         Y_side_batch[...,0] = batch_data[...,0]
#         Y_side_batch[...,1] = batch_data[...,3]
#         CbCr_batch = np.zeros((batch_size, image_size, image_size, 2), dtype = float)
#         CbCr_batch = batch_data[...,1:3]
#         yield (Y_side_batch, CbCr_batch)

#model.fit_generator(image_YCbCr_gen(batch_size), callbacks=[tensorboard], epochs=2, steps_per_epoch=batch_size)
# YCbCr_batch_data = YCbCr_training[...,0:3]
# side_batch_data = YCbCr_training[...,3]
# Y_side_batch = np.zeros((batch_size, image_size, image_size, 2), dtype = float)
# Y_side_batch[...,0] = YCbCr_training[...,0]
# Y_side_batch[...,1] = YCbCr_training[...,3]
# CbCr_batch = np.zeros((batch_size, image_size, image_size, 2), dtype = float)
# CbCr_batch = YCbCr_training[...,1:3]
# model.fit(x=Y_side_batch, y=CbCr_batch, batch_size=num_images, epochs=100)

# # Save model
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights("colourized_compression.h5")

# # TESTING IF OUTPUTS CORRECTLY
# color_me = np.zeros((image_size, image_size, 2), dtype = float)
# color_me[...,0] = YCbCr_training[0,:,:,0]
# color_me[...,1] = YCbCr_training[0,:,:,3]
# color_me = color_me.reshape(1, image_size, image_size, 2)
# output = model.predict(color_me)
# output = output * 255.0

# original_image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
# original_image[...,0] = YCbCr_training[0,:,:,0] * 255.0
# original_image[...,1:] = YCbCr_training[0,:,:,1:3] * 255.0
# original_image = Image.fromarray(original_image, mode = "YCbCr")
# original_image.show()

# new_image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
# new_image[...,0] = YCbCr_training[0,:,:,0] * 255.0
# new_image[...,1:] = output
# new_image = Image.fromarray(new_image, mode = "YCbCr")
# new_image.show()


# Fit and evaluate
cnn_x_image = images[0:num_images,:,:,0].reshape(num_images, image_size, image_size, 1)
cnn_x_side = side_information[0:num_images,:,:].reshape(num_images, image_size, image_size, 1)
cnn_y = images[0:num_images,:,:,1:].reshape(num_images, image_size, image_size, 2)
#model.fit(x=[cnn_x_image, cnn_x_side], y=cnn_y, batch_size=num_images, epochs=100)
#print(model.evaluate([cnn_x_image, cnn_x_side], cnn_y, batch_size=num_images))
model.fit(x=[cnn_x_image, cnn_x_side], y=cnn_y, epochs=10)
print(model.evaluate([cnn_x_image, cnn_x_side], cnn_y))

# Prediction
prediction_image = cnn_x_image[0,:,:,0].reshape(1, image_size, image_size, 1)
prediction_side = cnn_x_side[0,:,:].reshape(1, image_size, image_size, 1)
expected_output = cnn_y[0,:,:,:].reshape(1, image_size, image_size, 2)
cnn_output = model.predict([prediction_image, prediction_side])

# Output image versus original 
cnn_output = (cnn_output * 128.0) + 128.0
prediction_image = (prediction_image * 128.0) + 128.0
expected_output = (expected_output * 128.0) + 128.0
cnn_combined_image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
cnn_combined_image[..., 0] = prediction_image.reshape(image_size, image_size)
cnn_combined_image[..., 1:] = cnn_output.reshape(image_size, image_size, 2)
cnn_combined_image = Image.fromarray(np.uint8(cnn_combined_image), mode = "YCbCr")
cnn_combined_image.show()
original_image = np.zeros((image_size, image_size, 3), dtype = np.uint8)
original_image[..., 0] = prediction_image.reshape(image_size, image_size)
original_image[..., 1:] = expected_output.reshape(image_size, image_size, 2)
original_image = Image.fromarray(np.uint8(original_image), mode = "YCbCr")
original_image.show()














# #Make a prediction on the unseen images
# color_me = []
# for filename in os.listdir('../Test/'):
#     color_me.append(img_to_array(load_img('../Test/'+filename)))
# color_me = np.array(color_me, dtype=float)
# color_me = 1.0/255*color_me
# color_me = gray2rgb(rgb2gray(color_me))
# color_me = rgb2lab(color_me)[:,:,:,0]
# color_me = color_me.reshape(color_me.shape+(1,))

# # Test model
# output = model.predict(color_me)
# output = output * 128

# # Output colorizations
# for i in range(len(output)):
#     cur = np.zeros((256, 256, 3))
#     cur[:,:,0] = color_me[i][:,:,0]
#     cur[:,:,1:] = output[i]
#     imsave("result/img_"+str(i)+".png", lab2rgb(cur))













# DATA 
# NEURAL NETWORK

# image, side
# right side is 0.023330816 GB per input
# left side is 0.011522048 GB per input
# BOTH SIDES = 0.034852864 GB per input

# fusion,output
# 0.017956864 GB per input


# TOTAL = 0.052809728 GB per input
# http://cs231n.github.io/convolutional-networks/#case
# *2 everything for backward also, therefore total is 0.105619456 GB per input

# Roughly 6.6 million parameters * 3 * 4 = 0.0792 GB for parameters
# Total is now 0.184819456 + a bit GB per input? Round to 0.2GB per input?