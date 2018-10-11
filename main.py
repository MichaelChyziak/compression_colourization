# Python version: Python 2.7.15rc1
# Commands needed:
# sudo apt-get install python-numpy
# python -m pip install -U matplotlib
# pip install keras
# pip install --upgrade tensorflow  

import numpy as np
import matplotlib.pyplot as plt
import Tkinter as tk
import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from keras.models import Sequential

def main():
    grayscale_data_normalized = []
    dominant_colour_data = []
    colour_data_normalized = []
    cifar_data = getRGBCifarData("./dataset/cifar-10-batches-py/data_batch_1")
    for rgb_data in cifar_data:
        cnn_data = getCNNInputOutputData(rgb_data)
        grayscale_data_normalized.append(cnn_data["grayscale_data_normalized"])
        dominant_colour_data.append(cnn_data["dominant_colour_data"])
        colour_data_normalized.append(cnn_data["colour_data_normalized"])
    model = createCNNModel()

    cnn_input = np.array(grayscale_data_normalized, dtype = float)
    cnn_input = cnn_input.reshape(len(grayscale_data_normalized), 32, 32, 1)
    cnn_expected_output = np.array(colour_data_normalized, dtype = float)
    cnn_expected_output = cnn_expected_output.reshape(len(cnn_expected_output), 32, 32, 3)
    model.fit(x=cnn_input, y=cnn_expected_output, batch_size=5, epochs=200)

    print(model.evaluate(cnn_input, cnn_expected_output, batch_size=5))
    cnn_input_prediction = cnn_input[0].reshape(1, 32, 32, 1)
    cnn_output_normalized = model.predict(cnn_input_prediction)
    cnn_output = (cnn_output_normalized * 128 + 128)
    cnn_image_colour = cnn_output.reshape(32, 32, 3)
    cnn_image_colour = cnn_image_colour.astype(np.uint8)
    plt.figure()
    plt.imshow(cnn_image_colour)
    cnn_input_prediction_colour = (cnn_expected_output[0] * 128 + 128)
    cnn_input_prediction_colour = cnn_input_prediction_colour.reshape(32, 32, 3)
    cnn_input_prediction_colour = cnn_input_prediction_colour.astype(np.uint8)
    plt.figure()
    plt.imshow(cnn_input_prediction_colour)
    plt.show()

def createCNNModel():
    model = Sequential()
    model.add(InputLayer(input_shape=(32, 32, 1)))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
    model.compile(optimizer='rmsprop', loss='mse')
    return model

def getRGBCifarData(file_path):
    data = unpickle(file_path)
    print data["filenames"][0]
    return data["data"][0:10]

def getCNNInputOutputData(rgb_data):
    grayscale_data_normalized = np.empty([len(rgb_data) / 3], dtype = float)
    dominant_colour_data = []
    colour_data_normalized = np.empty([len(rgb_data) / 3, 3], dtype = float)
    for index in range(len(rgb_data) / 3):
        r_value = rgb_data[index]
        g_value = rgb_data[index + len(rgb_data) / 3];
        b_value = rgb_data[index + 2 * len(rgb_data) / 3];
        grayscale_data_normalized[index] = ((int(16.0 + (65.481 * r_value + 128.553 * g_value + 24.966 * b_value) / 256.0)) - 128.0) / 128.0
        colour_data_normalized[index] = [(r_value - 128.0) / 128.0, (g_value - 128.0) / 128.0, (b_value - 128.0) / 128.0]
        if r_value >= g_value and r_value >= b_value:
            dominant_colour_data.append("red")
        elif g_value >= r_value and g_value >= b_value:
            dominant_colour_data.append("green")
        elif b_value >= r_value and b_value >= g_value:
            dominant_colour_data.append("blue")
    return {"grayscale_data_normalized":grayscale_data_normalized, "colour_data_normalized":colour_data_normalized, "dominant_colour_data": dominant_colour_data}

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

if __name__ == "__main__":
    main()
