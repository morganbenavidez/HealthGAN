
import csv
import math
import time
import medmnist
import numpy as np
import tensorflow as tf
#from gan4 import train_gan
import matplotlib.pyplot as plt
from keras import layers, models
from neuralNetwork import cnn_kfold



def test_model(test_images, test_labels, model_name):
    loaded_model = models.load_model('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Models/' + model_name + '.h5')

    #predictions = loaded_model.predict(test_images)
    #print(predictions)
    evaluation = loaded_model.evaluate(test_images, test_labels)
    
    print(f"Test Loss: {evaluation[0]}, Test Accuracy: {evaluation[1]}")
    time.sleep(5)
    test_loss = evaluation[0]
    test_accuracy = evaluation[1]
    return test_loss, test_accuracy


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
    cnn_kfold(c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, number_of_classes, c_model_name, epochs, batch_size)

    # Begin training generated data Cardiomegaly network with balanced generated data
    number_of_classes = 1
    g_model_name = 'Initial_Cardiomegaly_Model2_generated_data'
    epochs = 500
    batch_size = 32
    cnn_kfold(new_balanced_train, new_balanced_train_labels, new_balanced_val, new_balanced_val_labels, number_of_classes, g_model_name, epochs, batch_size)

    #test_model(p_balanced_test, p_balanced_test_labels, p_model_name)
    test_model(c_balanced_test, c_balanced_test_labels, c_model_name)
    test_model(new_balanced_test, new_balanced_test_labels, g_model_name)
    #pneumonia_class_index, pneumonia_normal_class_index = get_pneumonia_indices(p_train_labels)
    

#main()

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

    #train_gan(c_train_images)

#gan_training()


def get_generated_data():
    batch_number = ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']
    #generated_data = [ '0','1', '2', '3', '4', '5', '6','7','8','9']
    all_completed_batches = []
    for i in range(0, len(batch_number)):
        ##for j in range(1, 4):
        train_images_1 = np.load('Data/generated_train_' + batch_number[i] + '_1.npz')['all_generated_images']
        train_images_2 = np.load('Data/generated_train_' + batch_number[i] + '_2.npz')['all_generated_images']
        train_images_3 = np.load('Data/generated_train_' + batch_number[i] + '_3.npz')['all_generated_images']
        print(train_images_1.shape)
        print(train_images_2.shape)
        print(train_images_3.shape)
        if train_images_3.shape == (0,):
            slice1 = train_images_1[:]
            slice2 = train_images_2[:]
            training_complete_batch = np.concatenate([slice1, slice2])
        else:
            slice1 = train_images_1[:]
            slice2 = train_images_2[:]
            slice3 = train_images_3[:]
            training_complete_batch = np.concatenate([slice1, slice2, slice3])

        
        all_completed_batches.append(training_complete_batch)
    
    return all_completed_batches

def graph_results():

    pass


def print_data(total_results, file_path, round, epochs2):

    with open(file_path, 'a') as file:
        file.write(f'{total_results}')

    for i in range(0, len(total_results)):
        print('\n')
        print(total_results[i][0][0] + ': ')
        print('Real Training Accuracy: ', total_results[i][1][0])
        print('Real Validation Loss: ', total_results[i][1][1])
        print('Real Validation Accuracy: ', total_results[i][1][2])
        print('Real Test Loss: ', total_results[i][1][3])
        print('Real Test Accuracy: ', total_results[i][1][4])
        print('Gen Training Accuracy: ', total_results[i][2][0])
        print('Gen Validation Loss: ', total_results[i][2][1])
        print('Gen Validation Accuracy: ', total_results[i][2][2])
        print('Gen Test Loss: ', total_results[i][2][3])
        print('Gen Test Accuracy: ', total_results[i][2][4])
    
        with open(file_path, 'a') as file:
            # Append content to the file
            file.write(f'Batch {total_results[i][0][0]}: ')
            file.write('\n')
            file.write(f'Real Training Accuracy: {total_results[i][1][0]}')
            file.write('\n')
            file.write(f'Real Validation Loss: {total_results[i][1][1]}')
            file.write('\n')
            file.write(f'Real Validation Accuracy: {total_results[i][1][2]}')
            file.write('\n')
            file.write('\n')
            file.write(f'Real Test Loss: {total_results[i][1][3]}')
            file.write('\n')
            file.write(f'Real Test Accuracy: {total_results[i][1][4]}')
            file.write('\n')
            file.write(f'Gen Training Accuracy: {total_results[i][2][0]}')
            file.write('\n')
            file.write('\n')
            file.write(f'Gen Validation Loss: {total_results[i][2][1]}')
            file.write('\n')
            file.write(f'Gen Validation Accuracy: {total_results[i][2][2]}')
            file.write('\n')
            file.write(f'Gen Test Loss: {total_results[i][2][3]}')
            file.write('\n')
            file.write(f'Gen Test Accuracy: {total_results[i][2][4]}')
            file.write('\n')

    # Open the CSV file in append mode
    with open('results.csv', 'a', newline='') as csvfile:
        # Create a CSV writer
        csvwriter = csv.writer(csvfile)

        # Write header if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(['Epochs', 'Batch', 'Real Training Accuracy', 'Real Validation Loss', 'Real Validation Accuracy',
                                'Real Test Loss', 'Real Test Accuracy', 'Gen Training Accuracy',
                                'Gen Validation Loss', 'Gen Validation Accuracy', 'Gen Test Loss', 'Gen Test Accuracy', 'Gen Epochs'])

        # Iterate through total_results and append the content to the CSV file
        for result in total_results:
            epochs = round
            batch_number = result[0][0]
            real_train_acc = result[1][0]
            real_val_loss = result[1][1]
            real_val_acc = result[1][2]
            real_test_loss = result[1][3]
            real_test_acc = result[1][4]
            gen_train_acc = result[2][0]
            gen_val_loss = result[2][1]
            gen_val_acc = result[2][2]
            gen_test_loss = result[2][3]
            gen_test_acc = result[2][4]
            epochs_gen = epochs2

            # Append the content to the CSV file
            csvwriter.writerow([epochs, batch_number, real_train_acc, real_val_loss, real_val_acc,
                                real_test_loss, real_test_acc, gen_train_acc,
                                gen_val_loss, gen_val_acc, gen_test_loss, gen_test_acc, epochs2])

def final_train():#batch_size, epochs, file_path, round):
    file_path = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_OneToRuleAll.txt'
    batch_size = 32
    # Declare data sets
    mnist_data = ['pneumoniamnist.npz', 'cardiomegaly.npz', 'normal_chest.npz', 'generated_images_train.npz', 'high_class_generated.npz', 'high_class_generated2.npz']

    # Get initial data set
    choice = mnist_data[0]
    p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels = prep_data(choice)

    choice2 = mnist_data[1]
    c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels = prep_data(choice2)

    choice3 = mnist_data[2]
    n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels = prep_data(choice3)
    
    #gd80 = generated_data[0]
    #eighty_to_eighty_nine = []
    
    # ['80', '81', '82', '83', '84', '85', '86', '87', '88', '89']
    print('all_completed_batches')
    all_completed_batches = get_generated_data()
    for j in range(0,len(all_completed_batches)):
        print(len(all_completed_batches[j]))

    #time.sleep(20)

    # Balance Cardiomegaly Sets
    c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, c_balanced_test, c_balanced_test_labels = balance_cardiomegaly_data(c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    # Balance Pneumonia Sets
    p_balanced_train, p_balanced_train_labels, p_balanced_val, p_balanced_val_labels, p_balanced_test, p_balanced_test_labels = balance_pneumonia_data(p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)

    c_gen_combined_batches = []

    c_train_slice_all = c_train_images[:]
    for i in range(0, len(all_completed_batches)):
        batch_slice = all_completed_batches[i][:]
        new_train_batch = np.concatenate([c_train_slice_all, batch_slice])
        np.random.shuffle(new_train_batch)
        c_gen_combined_batches.append(new_train_batch)
    print('\n')
    """
    for k in range(0, len(c_gen_combined_batches)):
        #print(len(c_gen_combined_batches[k]))
        new_train_images = c_gen_combined_batches[k]
        new_train_labels = np.array([[1] for _ in range(len(new_train_images))])
        new_balanced_train, new_balanced_train_labels, new_balanced_val, new_balanced_val_labels, new_balanced_test, new_balanced_test_labels = balance_cardiomegaly_data(new_train_images, new_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
        print(len(new_balanced_train))
    time.sleep(20)
    """
    with open(file_path, 'a') as file:
        file.write(f'c_val_images: {len(c_val_images)}')
        file.write('\n')
        file.write(f'c_test_images: {len(c_test_images)}')
        file.write('\n')
    total_results = []
    for j in range(0, len(c_gen_combined_batches)):
        print('Round: ', str(j))
        print('Round: ', str(j))
        print('Round: ', str(j))
        new_train_images = c_gen_combined_batches[j]
        new_train_labels = np.array([[1] for _ in range(len(new_train_images))])
        new_balanced_train, new_balanced_train_labels, new_balanced_val, new_balanced_val_labels, new_balanced_test, new_balanced_test_labels = balance_cardiomegaly_data(new_train_images, new_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
        if len(new_balanced_train) == 26730:
            round = 0
        elif len(new_balanced_train) == 22106:
            round = 1
        elif len(new_balanced_train) == 17600:
            round = 2
        elif len(new_balanced_train) == 13124:
            round = 3
        elif len(new_balanced_train) == 9712:
            round = 4
        elif len(new_balanced_train) == 7076:
            round = 5
        elif len(new_balanced_train) == 5302:
            round = 6
        elif len(new_balanced_train) == 4380:
            round = 7
        elif len(new_balanced_train) == 4032:
            round = 8
        elif len(new_balanced_train) == 3934:
            round = 9
        print('ZZZZ')
        print(len(c_balanced_train))
        print(len(c_balanced_val))
        print(len(c_balanced_test))
        #time.sleep(20)
        print(len(new_balanced_train))
        #time.sleep(20)
        # Begin training initial Cardiomegaly network with balanced data
        number_of_classes = 1
        c_model_name = 'Initial_Cardiomegaly_real_data' + '_' + str(j) + '_' + str(round)
        epochs = 40
        #batch_size = batch_size
        training_accuracy_real, validation_loss_real, validation_accuracy_real = cnn_kfold(c_balanced_train, c_balanced_train_labels, c_balanced_val, c_balanced_val_labels, number_of_classes, c_model_name, epochs, batch_size)

        # Begin training generated data Cardiomegaly network with balanced generated data
        number_of_classes = 1
        g_model_name = 'Initial_Cardiomegaly_generated_data' + '_' + str(j) + '_' + str(round)
        
        if len(new_balanced_train) > 20000:
            epochs2 = 200
        elif len(new_balanced_train) > 10000:
            epochs2 = 100
        elif len(new_balanced_train) > 5000:
            epochs2 = 80
        else:
            epochs2 = 40

        #epochs = 400
        #batch_size = 32
        training_accuracy_gen, validation_loss_gen, validation_accuracy_gen = cnn_kfold(new_balanced_train, new_balanced_train_labels, new_balanced_val, new_balanced_val_labels, number_of_classes, g_model_name, epochs2, batch_size)

        #test_model(p_balanced_test, p_balanced_test_labels, p_model_name)
        test_loss_real, test_accuracy_real = test_model(c_balanced_test, c_balanced_test_labels, c_model_name)
        test_loss_gen, test_accuracy_gen = test_model(new_balanced_test, new_balanced_test_labels, g_model_name)

        real_results = [training_accuracy_real, validation_loss_real, validation_accuracy_real, test_loss_real, test_accuracy_real]
        gen_results = [training_accuracy_gen, validation_loss_gen, validation_accuracy_gen, test_loss_gen, test_accuracy_gen]

        batch_results = [[str(j)], real_results, gen_results]
        total_results.append(batch_results)

    print_data(total_results, file_path, round, epochs2)


#final_train()


def print_data_pneumonia(total_results, file_path):

    with open(file_path, 'a') as file:
        file.write(f'{total_results}')

    for i in range(0, len(total_results)):
        print('\n')
        print(total_results[i][0][0] + ': ')
        print('Real Training Accuracy: ', total_results[i][1][0])
        print('Real Validation Loss: ', total_results[i][1][1])
        print('Real Validation Accuracy: ', total_results[i][1][2])
        print('Real Test Loss: ', total_results[i][1][3])
        print('Real Test Accuracy: ', total_results[i][1][4])
       
    
        with open(file_path, 'a') as file:
            # Append content to the file
            file.write(f'Batch {total_results[i][0][0]}: ')
            file.write('\n')
            file.write(f'Real Training Accuracy: {total_results[i][1][0]}')
            file.write('\n')
            file.write(f'Real Validation Loss: {total_results[i][1][1]}')
            file.write('\n')
            file.write(f'Real Validation Accuracy: {total_results[i][1][2]}')
            file.write('\n')
            file.write('\n')
            file.write(f'Real Test Loss: {total_results[i][1][3]}')
            file.write('\n')
            file.write(f'Real Test Accuracy: {total_results[i][1][4]}')
            file.write('\n')

    # Open the CSV file in append mode
    with open('results_pneumonia.csv', 'a', newline='') as csvfile:
        # Create a CSV writer
        csvwriter = csv.writer(csvfile)

        # Write header if the file is empty
        if csvfile.tell() == 0:
            csvwriter.writerow(['Batch', 'Real Training Accuracy', 'Real Validation Loss', 'Real Validation Accuracy',
                                'Real Test Loss', 'Real Test Accuracy'])

        # Iterate through total_results and append the content to the CSV file
        for result in total_results:
            batch_number = result[0][0]
            real_train_acc = result[1][0]
            real_val_loss = result[1][1]
            real_val_acc = result[1][2]
            real_test_loss = result[1][3]
            real_test_acc = result[1][4]


            # Append the content to the CSV file
            csvwriter.writerow([batch_number, real_train_acc, real_val_loss, real_val_acc,
                                real_test_loss, real_test_acc])


def pneumonia_only_test():
    file_path = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_pneumonia.txt'
    batch_size = 32
    # Declare data sets
    mnist_data = ['pneumoniamnist.npz', 'cardiomegaly.npz', 'normal_chest.npz', 'generated_images_train.npz', 'high_class_generated.npz', 'high_class_generated2.npz']

    # Get initial data set
    choice = mnist_data[0]
    p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels = prep_data(choice)

    choice3 = mnist_data[2]
    n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels = prep_data(choice3)
    
    # Balance Pneumonia Sets
    p_balanced_train, p_balanced_train_labels, p_balanced_val, p_balanced_val_labels, p_balanced_test, p_balanced_test_labels = balance_pneumonia_data(p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels, n_train_images, n_train_labels, n_val_images, n_val_labels, n_test_images, n_test_labels)
    print(len(p_balanced_train))
    print(len(p_balanced_val))
    print(len(p_balanced_test))
    time.sleep(20)
    total_results = []
    for j in range(0, 10):
        print('Round: ', str(j))
        print('Round: ', str(j))
        print('Round: ', str(j))
        
        # Begin training initial Cardiomegaly network with balanced data
        number_of_classes = 1
        p_model_name = 'Initial_Cardiomegaly_pneumonia_data' + '_' + str(j) + '_' + str(round)
        epochs = 40
        #batch_size = batch_size
        training_accuracy_real, validation_loss_real, validation_accuracy_real = cnn_kfold(p_balanced_train, p_balanced_train_labels, p_balanced_val, p_balanced_val_labels, number_of_classes, p_model_name, epochs, batch_size)

        #test_model(p_balanced_test, p_balanced_test_labels, p_model_name)
        test_loss_real, test_accuracy_real = test_model(p_balanced_test, p_balanced_test_labels, p_model_name)

        real_results = [training_accuracy_real, validation_loss_real, validation_accuracy_real, test_loss_real, test_accuracy_real]

        batch_results = [[str(j)], real_results]
        total_results.append(batch_results)

    print_data_pneumonia(total_results, file_path)

pneumonia_only_test()
"""
file_path_100 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_100.txt'  # Replace with your file path
file_path_150 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_150.txt'  # Replace with your file path
file_path_200 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_200.txt'  # Replace with your file path
file_path_250 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_250.txt'  # Replace with your file path
file_path_300 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_300.txt'  # Replace with your file path
file_path_350 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_350.txt'  # Replace with your file path
file_path_400 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_400.txt'  # Replace with your file path
file_path_450 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_450.txt'  # Replace with your file path
file_path_500 = '/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Results/results_500.txt'  # Replace with your file path

file_paths = [file_path_100, file_path_150, file_path_200, file_path_250, file_path_300, file_path_350, file_path_400, file_path_450, file_path_500]
#epochs = [100, 150, 200, 250, 300, 350, 400, 450, 500]
epoch_records = [20, 30, 40, 50, 60, 70, 80, 90, 100]
# Record start time
start_time = time.time()
for i in range(1, len(file_paths)):
    batch_size = 32
    epochs = epoch_records[i]
    round = str(epoch_records[i])
    file_path = file_paths[i]
    start_time_inner = time.time()
    final_train(batch_size, epochs, file_path, round)
    end_time_inner = time.time()
    elapsed_time_inner = end_time_inner - start_time_inner
    print(f"Elapsed inner time: {elapsed_time_inner} seconds")
# Record end time
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")

"""
    