import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import pickle
import time
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pandas as pd
import csv
import json
import datetime


# funzione per creare il modello
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


# funzione per caricare i dati
def load_data():
    # carica i dati
    X = pickle.load(open("X.pickle", "rb"))
    y = pickle.load(open("y.pickle", "rb"))
    X = X / 255.0
    return X, y


# funzione per salvare il modello
def save_model(model):
    model.save('modello.h5')


# funzione per caricare il modello
def load_model():
    model = keras.models.load_model('modello.h5')
    return model


# funzione per creare il grafico
def create_graph(history):
    # grafico
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


# funzione per creare il grafico
def create_graph_loss(history):
    # grafico
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()



def main():
    # carica i dati
    X, y = load_data()

    # crea il modello
    model = create_model()

    # stampa il modello
    print(model.summary())

    # esegue il training
    history = model.fit(X, y, batch_size=32, epochs=5, validation_split=0.1)

    # salva il modello
    save_model(model)

    # crea il grafico
    create_graph(history)

    # crea il grafico
    create_graph_loss(history)


if __name__ == "__main__":
    main()
    
