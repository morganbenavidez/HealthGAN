

import time
import medmnist
import numpy as np
import tensorflow as tf
from keras import layers, models
#from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import math

def cnn(train_images, train_labels, val_images, val_labels, num_classes, model_name, epoch_number, batch_size_number):

    # Define the CNN model
    model = models.Sequential()
    """
    # Convolutional Layers
    model.add(layers.Conv2D(2, (3, 3), activation='relu', input_shape=(28, 28, 1))) # Changed from 32

    model.add(layers.Dropout(0.5))  # Add dropout layer

    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(num_classes, activation="sigmoid"))
    """
    #model.add(layers.Conv2D(1, (2, 2), activation='relu', input_shape=(28, 28, 1))) # Changed from 32

    #model.add(layers.Dropout(0.5))  # Add dropout layer

    #model.add(layers.MaxPooling2D((2, 2)))

    # Got rid of these layers
    #model.add(layers.Conv2D(8, (3, 3), activation='relu')) # Changed from 64 to 32
    #model.add(layers.MaxPooling2D((2, 2)))
    
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))

    # Flatten Layer
    
    # Fully Connected Layers with Dropout
    #model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dropout(0.5))  # Add dropout with a 50% dropout rate

    #model.add(layers.Dense(2, activation='relu')) # Changed from 64

    

    #model.add(layers.Dropout(0.5))  # Add another dropout layer
    
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer to transition from convolutional to dense layers
    model.add(layers.Flatten())

    # Dense (fully connected) layers with dropout
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout layer to prevent overfitting
    #model.add(layers.Dense(10, activation='softmax'))
    model.add(layers.Dense(num_classes, activation="sigmoid"))


    """
    # Add a convolutional layer with 32 filters, each 3x3 in size, and ReLU activation function
    model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    #model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    #model.add(layers.MaxPooling2D((2, 2)))

    # Fully Connected Layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dropout(0.5))  # Regularization using dropout
    model.add(layers.Dense(num_classes, activation='softmax'))  # num_classes is the number of classes
    
    # Add a max-pooling layer with 2x2 pool size
    model.add(layers.MaxPooling2D((2, 2)))

    # Add another convolutional layer with 64 filters, each 3x3 in size, and ReLU activation function
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add another convolutional layer with 64 filters, each 3x3 in size, and ReLU activation function
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))

    # Add a max-pooling layer with 2x2 pool size
    model.add(layers.MaxPooling2D((2, 2)))

    # Add another convolutional layer with 64 filters, each 3x3 in size, and ReLU activation function
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))

    # Flatten the output to a 1D tensor
    model.add(layers.Flatten())

    # Add a fully connected (dense) layer with 64 units and ReLU activation
    model.add(layers.Dense(96, activation='relu'))

    # Add a fully connected (dense) layer with 64 units and ReLU activation
    model.add(layers.Dense(64, activation='relu'))

    # Add the output layer with as many units as the number of classes, and softmax activation function for classification
    model.add(layers.Dense(num_classes, activation="sigmoid"))
    """
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model architecture
    model.summary()

    history = model.fit(
        train_images,
        train_labels,
        epochs=epoch_number,
        batch_size=batch_size_number,
        validation_data=(val_images, val_labels)
    )
    print(history)

    model.save('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Models/' + model_name + '.h5')

    validation_loss, validation_accuracy = model.evaluate(val_images, val_labels)
    print('Validation loss: ', validation_loss)
    print('Validation accuracy: ', validation_accuracy)

    return model