import time
import medmnist
import numpy as np
import tensorflow as tf
from keras import layers, models
#from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import StratifiedKFold


def cnn_kfold(train_images, train_labels, val_images, val_labels, num_classes, model_name, epoch_number, batch_size_number, n_splits=3):
    # Combine training and validation sets for k-fold cross-validation
    slice1 = train_images[:]
    slice2 = val_images[:]
    slice3 = train_labels[:]
    slice4 = val_labels[:]
    X = np.concatenate([slice1, slice2])
    #print(X)
    y = np.concatenate([slice3, slice4])
    print(y)
    #time.sleep(10)
    #X = np.concatenate([train_images, val_images], axis=0)
    #y = np.concatenate([train_labels, val_labels], axis=0)
    #np.random.shuffle(X)
    #np.random.shuffle(y)

    # Define the CNN model
    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(num_classes, activation='sigmoid'))
    """
    # Used to start with 4 and go to 8 and 16 dense layers
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same', input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    # Flatten layer to transition from convolutional to dense layers
    model.add(layers.Flatten())
    # Dense (fully connected) layers with dropout
    model.add(layers.Dense(16, activation='relu')) # Changed from 128 to 64
    model.add(layers.Dropout(0.5)) # Dropout layer to prevent overfitting
    #model.add(layers.Dense(10, activation='softmax'))
    model.add(layers.Dense(num_classes, activation="sigmoid"))
    """
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Summary of the model architecture
    model.summary()

    # Perform k-fold cross-validation
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    training_accuracies = []
    validation_accuracies = []

    for train_index, val_index in kfold.split(X, y):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        history = model.fit(
            X_train,
            y_train,
            epochs=epoch_number,
            batch_size=batch_size_number,
            validation_data=(X_val, y_val)
        )

        training_accuracies.append(history.history['accuracy'][-1])
        validation_loss, validation_accuracy = model.evaluate(X_val, y_val)
        validation_accuracies.append(validation_accuracy)

    # Calculate average accuracy across folds
    avg_training_accuracy = np.mean([float(acc) for acc in training_accuracies])
    avg_validation_accuracy = np.mean([float(acc) for acc in validation_accuracies])

    print('Average Training Accuracy: ', avg_training_accuracy)
    print('Average Validation Accuracy: ', avg_validation_accuracy)

    model.save('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Models/' + model_name + '.h5')

    return avg_training_accuracy, validation_loss, avg_validation_accuracy