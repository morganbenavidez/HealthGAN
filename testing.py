import math
import time
import medmnist
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
from neuralNetwork import cnn
from gan3 import train_gan


def test_model(test_images, test_labels, model_name):
    loaded_model = models.load_model('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Models/' + model_name + '.h5')

    #predictions = loaded_model.predict(test_images)
    #print(predictions)
    evaluation = loaded_model.evaluate(test_images, test_labels)
    print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")


# Get groups
def get_groups(data):
    # ['train_images', 'val_images', 'test_images', 'train_labels', 'val_labels', 'test_labels']
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

# Find the files in the numpy data
def find_files_in_numpy_data(data):
    files = data.files
    return files

# Load data
def load_data(choice):
    if choice == 'pneumoniamnist.npz': 
        data = np.load(choice)
        train_images, train_labels, val_images, val_labels, test_images, test_labels = get_groups(data)
        return train_images, train_labels, val_images, val_labels, test_images, test_labels
    
    elif choice == 'cardiomegaly.npz':

        train_images = np.load('Chest/cardiomegaly_images_train.npz')['cardiomegaly_data']
        val_images = np.load('Chest/cardiomegaly_images_val.npz')['cardiomegaly_data']
        test_images = np.load('Chest/cardiomegaly_images_test.npz')['cardiomegaly_data']

        #train_labels = np.full(len(train_images), fill_value=[1])
        train_labels = np.array([[1] for _ in range(len(train_images))])
        #val_labels = np.full(len(val_images), fill_value=[1])
        val_labels = np.array([[1] for _ in range(len(val_images))])
        #test_labels = np.full(len(test_images), fill_value=[1])
        test_labels = np.array([[1] for _ in range(len(test_images))])

        return train_images, train_labels, val_images, val_labels, test_images, test_labels
    
    elif choice == 'normal_chest.npz':
        train_images = np.load('Chest/normal_chest_images_train.npz')['normal_data']
        val_images = np.load('Chest/normal_chest_images_val.npz')['normal_data']
        test_images = np.load('Chest/normal_chest_images_test.npz')['normal_data']

        #train_labels = np.full(len(train_images), fill_value=[1])
        train_labels = np.array([[0] for _ in range(len(train_images))])
        #val_labels = np.full(len(val_images), fill_value=[1])
        val_labels = np.array([[0] for _ in range(len(val_images))])
        #test_labels = np.full(len(test_images), fill_value=[1])
        test_labels = np.array([[0] for _ in range(len(test_images))])

        return train_images, train_labels, val_images, val_labels, test_images, test_labels
    
    elif choice == 'generated_images_train.npz':
        train_images = np.load('Chest/generated_images_train.npz')['all_generated_images']
        val_images = []
        test_images = []

        #train_labels = np.full(len(train_images), fill_value=[1])
        train_labels = np.array([[1] for _ in range(len(train_images))])
        #val_labels = np.full(len(val_images), fill_value=[1])
        val_labels = []
        #test_labels = np.full(len(test_images), fill_value=[1])
        test_labels = []

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    elif choice == 'high_class_generated.npz':
        train_images = np.load('Chest/high_class_generated.npz')['all_generated_images']
        val_images = []
        test_images = []

        #train_labels = np.full(len(train_images), fill_value=[1])
        train_labels = np.array([[1] for _ in range(len(train_images))])
        #val_labels = np.full(len(val_images), fill_value=[1])
        val_labels = []
        #test_labels = np.full(len(test_images), fill_value=[1])
        test_labels = []

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    elif choice == 'high_class_generated2.npz':
        train_images = np.load('Chest/high_class_generated2.npz')['all_generated_images']
        val_images = []
        test_images = []

        #train_labels = np.full(len(train_images), fill_value=[1])
        train_labels = np.array([[1] for _ in range(len(train_images))])
        #val_labels = np.full(len(val_images), fill_value=[1])
        val_labels = []
        #test_labels = np.full(len(test_images), fill_value=[1])
        test_labels = []

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    

# Prepare data
def prep_data(choice):
    
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(choice)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels

def separate_pneumonia():

    pass

def get_pneumonia_indices(p_train_images, p_train_labels):

    classes = [0, 0]
    pneumonia_class_index = []
    pneumonia_normal_class_index = []
    pneumonia_data = []
    normal_data = []

    for i in range(0, len(p_train_labels)):
        #print(p_train_labels[i])
        # normal
        if p_train_labels[i] == [0]:
            classes[0] = classes[0] + 1
            pneumonia_normal_class_index.append(i)
            normal_data.append(p_train_images[i])
        # pneumonia
        else:
            classes[1] = classes[1] + 1
            pneumonia_class_index.append(i)
            pneumonia_data.append(p_train_images[i])
    print(classes)
    normal_data_array = np.array(normal_data)
    pneumonia_data_array = np.array(pneumonia_data)
    return pneumonia_data_array, normal_data_array
    #return pneumonia_class_index, pneumonia_normal_class_index

def balance_pneumonia_data(p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels):

    print('TRAIN')
    pneumonia_class_train, pneumonia_normal_class_train = get_pneumonia_indices(p_train_images, p_train_labels)
    #print(len(pneumonia_class_train))
    #print(len(pneumonia_normal_class_train))
    # Here get length of pneumonia class
    # Get length of normal class
    # If pc is greater than len(normal) -> Add the difference from n_train etc to normal class
    # Generate the labels for both and return
    pc_train_length = len(pneumonia_class_train)
    n_train_length = len(pneumonia_normal_class_train)
    difference_train = pc_train_length - n_train_length
    p_normal_train = np.concatenate((pneumonia_normal_class_train, n_train_images[:difference_train]))
    print(len(p_normal_train))
    print(len(pneumonia_class_train))
    balanced_train = np.concatenate((pneumonia_class_train, p_normal_train))
    #print(len(pneumonia_class_train) + len(p_normal_train))
    print(len(balanced_train))
    #print(len(p_train_labels))
    balanced_train_labels = np.concatenate((np.array([[1] for _ in range(len(pneumonia_class_train))]), np.array([[0] for _ in range(len(p_normal_train))])))
    print(len(balanced_train_labels))
    
    #time.sleep(10)
    print('VAL')
    pneumonia_class_val, pneumonia_normal_class_val = get_pneumonia_indices(p_val_images, p_val_labels)
    #print(len(pneumonia_class_val))
    #print(len(pneumonia_normal_class_val))
    pc_val_length = len(pneumonia_class_val)
    n_val_length = len(pneumonia_normal_class_val)
    difference_val = pc_val_length - n_val_length
    p_normal_val = np.concatenate((pneumonia_normal_class_val, n_val_images[:difference_val]))
    print(len(p_normal_val))
    print(len(pneumonia_class_val))
    balanced_val = np.concatenate((pneumonia_class_val, p_normal_val))
    #print(len(pneumonia_class_val) + len(p_normal_val))
    print(len(balanced_val))
    balanced_val_labels = np.concatenate((np.array([[1] for _ in range(len(pneumonia_class_val))]), np.array([[0] for _ in range(len(p_normal_val))])))
    print(len(balanced_val_labels))

    
    #time.sleep(10)
    print('TEST')
    pneumonia_class_test, pneumonia_normal_class_test = get_pneumonia_indices(p_test_images, p_test_labels)
    #print(len(pneumonia_class_test))
    #print(len(pneumonia_normal_class_test))
    pc_test_length = len(pneumonia_class_test)
    n_test_length = len(pneumonia_normal_class_test)
    difference_test = pc_test_length - n_test_length
    p_normal_test = np.concatenate((pneumonia_normal_class_test, n_test_images[:difference_test]))
    print(len(p_normal_test))
    print(len(pneumonia_class_test))

    balanced_test = np.concatenate((pneumonia_class_test, p_normal_test))
    #print(len(pneumonia_class_test) + len(p_normal_test))
    print(len(balanced_test))
    balanced_test_labels = np.concatenate((np.array([[1] for _ in range(len(pneumonia_class_test))]), np.array([[0] for _ in range(len(p_normal_test))])))
    print(len(balanced_test_labels))


    #time.sleep(10)
    #separate_pneumonia()
    return balanced_train, balanced_train_labels, balanced_val, balanced_val_labels, balanced_test, balanced_test_labels



def balance_cardiomegaly_data(c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels):

     # Take equal parts from each array
    train_length = len(c_train_images)
    val_length = len(c_val_images)
    test_length = len(c_test_images)

    balanced_train = np.concatenate((c_train_images, n_train_images[:train_length]), axis=0)
    #print(c_train_labels[:10])
    #print(n_train_labels[:10])
    #time.sleep(10)
    #balanced_train_labels = np.concatenate(c_train_labels[0], n_train_labels[0][:train_length])
    balanced_train_labels = np.concatenate((c_train_labels, n_train_labels[:train_length]))
    #print(train_length)
    #print(train_length*2)
    #print(len(balanced_train))

    balanced_val = np.concatenate((c_val_images, n_val_images[:val_length]), axis=0)
    #balanced_val_labels = np.concatenate(c_val_labels, n_val_labels[:val_length])
    balanced_val_labels = np.concatenate((c_val_labels, n_val_labels[:val_length]))
    #print(val_length)
    #print(val_length*2)
    #print(len(balanced_val))

    balanced_test = np.concatenate((c_test_images, n_test_images[:test_length]), axis=0)
    #balanced_test_labels = np.concatenate(c_test_labels, n_test_labels[:test_length])
    balanced_test_labels = np.concatenate((c_test_labels, n_test_labels[:test_length]))
    #print(test_length)
    #print(test_length*2)
    #print(len(balanced_test))

    return balanced_train, balanced_train_labels, balanced_val, balanced_val_labels, balanced_test, balanced_test_labels



def main():
    
    # Declare data sets
    mnist_data = ['pneumoniamnist.npz', 'cardiomegaly.npz', 'normal_chest.npz', 'generated_images_train.npz', 'high_class_generated.npz', 'high_class_generated2.npz']

    # Get initial data set
    choice = mnist_data[0]
    p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels = prep_data(choice)

    choice2 = mnist_data[1]
    c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels = prep_data(choice2)

    choice3 = mnist_data[2]
    n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels = prep_data(choice3)
    
    choice4 = mnist_data[4]
    g_train_images, g_train_labels, g_val_images, g_val_labels, g_test_images, g_test_labels = prep_data(choice4)
    
    choice5 = mnist_data[5]
    g_train_images2, g_train_labels2, g_val_images2, g_val_labels2, g_test_images2, g_test_labels2 = prep_data(choice5)

    #print('p_labels: ', p_train_labels[0:10])
    #print('c_labels: ', c_train_labels[0:10])
    #print('n_labels: ', n_train_labels[0:10])
    slice1 = c_train_images[:]
    slice2 = g_train_images[:]
    new_train_images = np.concatenate([slice1, slice2])
    np.random.shuffle(new_train_images)
    new_train_labels = np.array([[1] for _ in range(len(new_train_images))])

    slice1 = new_train_images[:]
    slice2 = g_train_images2[:]

    new_train_images = np.concatenate([slice1, slice2])
    np.random.shuffle(new_train_images)
    new_train_labels = np.array([[1] for _ in range(len(new_train_images))])
    # Balance Cardiomegaly Sets
    # Uses generated cardiomegaly Data
    #c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, c_balanced_test, c_balanced_test_labels = balance_cardiomegaly_data(g_train_images, g_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    # Pure cardiomegaly Data
    c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, c_balanced_test, c_balanced_test_labels = balance_cardiomegaly_data(c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    
    print(len(c_balanced_train))
    print(len(c_balanced_train_labels))
    
    #c_train_images = c_train_images[:1000]
    #c_train_labels = c_train_labels[:1000]

    c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, c_balanced_test, c_balanced_test_labels = balance_cardiomegaly_data(c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)

    


    #print(c_balanced_train[:5])
    #new_train_images = new_train_images[:1000]
    #new_train_labels = new_train_labels[:1000]
    
    new_balanced_train, new_balanced_train_labels, new_balanced_val, new_balanced_val_labels, new_balanced_test, new_balanced_test_labels = balance_cardiomegaly_data(new_train_images, new_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    print('new balance')
    print(new_balanced_train[:5])
    print(len(c_balanced_train))
    print(len(c_balanced_train_labels))
    print(len(new_balanced_train))
    print(len(new_balanced_train_labels))
    #time.sleep(20)
    


    #time.sleep(10)
    print('\n')
    print(len(c_balanced_train))
    print(len(c_balanced_train_labels))
    print(len(c_balanced_val))
    print(len(c_balanced_val_labels))
    print(len(c_balanced_test))
    print(len(c_balanced_test_labels))
    print('\n')
    print(len(new_balanced_train))
    print(len(new_balanced_train_labels))
    print(len(new_balanced_val))
    print(len(new_balanced_val_labels))
    print(len(new_balanced_test))
    print(len(new_balanced_test_labels))
    print('\n')
    #time.sleep(20)
    # Balance Pneumonia Sets 
    p_balanced_train, p_balanced_train_labels, p_balanced_val, p_balanced_val_labels, p_balanced_test, p_balanced_test_labels = balance_pneumonia_data(p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    """
    print(len(p_balanced_train))
    print(len(p_balanced_val))
    print(len(p_balanced_test))
    print(len(p_balanced_train_labels))
    print(len(p_balanced_val_labels))
    print(len(p_balanced_test_labels))
    
    p_balanced_train = p_balanced_train[:3900]
    p_balanced_train_labels = p_balanced_train_labels[:3900]
    p_balanced_val = p_balanced_val[:480]
    p_balanced_val_labels = p_balanced_val_labels[:480]
    p_balanced_test = p_balanced_test[:1164]
    p_balanced_test_labels = p_balanced_test_labels[:1164]
    # Begin training initial Pneumonia network with balanced data
    number_of_classes = 1
    p_model_name = 'Initial_Pneumonia_Model_generated'
    epochs = 50
    batch_size = 32
    cnn(p_balanced_train, p_balanced_train_labels, p_balanced_val, p_balanced_val_labels, number_of_classes, p_model_name, epochs, batch_size)
    """
    # Begin training initial Cardiomegaly network with balanced data
    number_of_classes = 1
    c_model_name = 'Initial_Cardiomegaly_Model2_real_data'
    epochs = 500
    batch_size = 32
    cnn(c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, number_of_classes, c_model_name, epochs, batch_size)

    # Begin training generated data Cardiomegaly network with balanced generated data
    number_of_classes = 1
    g_model_name = 'Initial_Cardiomegaly_Model2_generated_data'
    epochs = 500
    batch_size = 32
    cnn(new_balanced_train, new_balanced_train_labels, new_balanced_val, new_balanced_val_labels, number_of_classes, g_model_name, epochs, batch_size)

    #test_model(p_balanced_test, p_balanced_test_labels, p_model_name)
    test_model(c_balanced_test, c_balanced_test_labels, c_model_name)
    test_model(new_balanced_test, new_balanced_test_labels, g_model_name)
    #pneumonia_class_index, pneumonia_normal_class_index = get_pneumonia_indices(p_train_labels)
    

main()

def pure_testing():

    # Declare data sets
    mnist_data = ['pneumoniamnist.npz', 'cardiomegaly.npz', 'normal_chest.npz']

    # Get initial data sets
    choice = mnist_data[0]
    p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels = prep_data(choice)

    choice2 = mnist_data[1]
    c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels = prep_data(choice2)

    choice3 = mnist_data[2]
    n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels = prep_data(choice3)
    #print('p_labels: ', p_train_labels[0:10])
    #print('c_labels: ', c_train_labels[0:10])
    #print('n_labels: ', n_train_labels[0:10])

    # Balance Cardiomegaly Sets
    c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, c_balanced_test, c_balanced_test_labels = balance_cardiomegaly_data(c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    print('CARDIOMEGALY DATA')
    print(len(c_balanced_train))
    print(len(c_balanced_train_labels))
    print(len(c_balanced_val))
    print(len(c_balanced_val_labels))
    print(len(c_balanced_test))
    print(len(c_balanced_test_labels))
    print('\n')
    
    # Balance Pneumonia Sets 
    p_balanced_train, p_balanced_train_labels, p_balanced_val, p_balanced_val_labels, p_balanced_test, p_balanced_test_labels = balance_pneumonia_data(p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    print('PNEUMONIA DATA')
    print(len(p_balanced_train))
    print(len(p_balanced_train_labels))
    print(len(p_balanced_val))
    print(len(p_balanced_val_labels))
    print(len(p_balanced_test))
    print(len(p_balanced_test_labels))

    
    
    p_model_name = 'Initial_Pneumonia_Model'
    c_model_name = 'Initial_Cardiomegaly_Model'

    print('TESTING MODELS')
    time.sleep(20)

    test_model(p_balanced_test, p_balanced_test_labels, p_model_name)
    test_model(c_balanced_test, c_balanced_test_labels, c_model_name)

#pure_testing()

def gan_training():

    # Declare data sets
    mnist_data = ['pneumoniamnist.npz', 'cardiomegaly.npz', 'normal_chest.npz']

    # Get initial data sets
    #choice = mnist_data[0]
    #p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels = prep_data(choice)

    choice2 = mnist_data[1]
    c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels = prep_data(choice2)
    print(len(c_train_images))

    train_gan(c_train_images)

#gan_training()