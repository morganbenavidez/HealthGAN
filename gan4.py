
import os
import PIL
import time
import glob
import imageio
import numpy as np
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from keras import layers, models

all_generated_images = []

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
    try:
        axes[0].imshow(generated_image[0, :, :, 0], cmap='gray')
    except:
        axes[0].imshow(generated_image, cmap='gray')
    
    axes[0].set_title('Generated Image')
    axes[0].axis('off')
    plt.show()
    #plt.imshow(, cmap='gray')


def make_generator_model():

    #model = tf.keras.Sequential()
    model = models.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None, 7, 7, 256) # None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    if epoch % 20 == 0:
        fig = plt.figure(figsize=(4, 4))
        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i+1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
        plt.savefig('GeneratedImages/image_at_epoch_{:04d}.png'.format(epoch))
    if epoch > 1000:
        for i in range(predictions.shape[0]):
            prediction = predictions[i, :, :, 0] * 127.5 + 127.5
            print(prediction)
            all_generated_images.append(prediction)
            #show_image(prediction)
            #time.sleep(10)
        #img = Image.fromarray(denormalized_good_image, 'L')
        #img.save(f'/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/GeneratedImages/good_generated_image_epoch_{epoch}_idx_{i}.png')
        #all_generated_images.append(denormalized_good_image)
    #if epoch % 10 == 0:
        #plt.show()

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


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
# Reshape
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
print(train_images.shape)
# Normalize to [-1,1]
train_images = (train_images - 127.5) /127.5

BUFFER_SIZE = 1950
BATCH_SIZE = 32

train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator = make_generator_model()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

show_image(generated_image)

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)


EPOCHS = 2000
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

train(train_dataset, EPOCHS)
    
all_generated_images_array = np.array(all_generated_images)
# Save the array of all generated images to an .npz file
np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Chest/generated_images_train_gan4.npz', all_generated_images=all_generated_images_array)
print(len(all_generated_images_array))