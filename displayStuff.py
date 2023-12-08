
import numpy as np
from IPython import display
import matplotlib.pyplot as plt


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
    
    elif choice == 'generated_images.npz':
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


# Prepare data
def prep_data(choice):
    
    train_images, train_labels, val_images, val_labels, test_images, test_labels = load_data(choice)
    return train_images, train_labels, val_images, val_labels, test_images, test_labels




def show_image(generated_image):

    # Change second number to increase columns on figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot real image
    axes[0].imshow(generated_image[0, :, :, 0], cmap='gray')
    axes[0].set_title('Generated Image')
    axes[0].axis('off')
    plt.show()
    #plt.imshow(, cmap='gray')

def show_multiple_images(predictions):

    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')#* 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    plt.show()

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    if epoch % 5 == 0:
        plt.savefig('GeneratedImages/image_at_epoch_{:04d}.png'.format(epoch))
    if epoch % 10 == 0:
        plt.show()

# Declare data sets
mnist_data = ['pneumoniamnist.npz', 'cardiomegaly.npz', 'normal_chest.npz']

# Get initial data sets
#choice = mnist_data[0]
#p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels = prep_data(choice)

choice2 = mnist_data[1]
c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels = prep_data(choice2)
print(len(c_train_images))

train_images = c_train_images

print(train_images.shape)

# Add a channel dimension
train_images = np.expand_dims(train_images, axis=-1)

# Now, the shape of data_with_channel is (1950, 28, 28, 1)
print(train_images.shape)
x = train_images[:16]
print(x.shape)
show_multiple_images(x)