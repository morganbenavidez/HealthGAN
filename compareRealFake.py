
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from testing import prep_data
import numpy as np
import time




def compare_structure(real_image, generated_image):
    # Assuming real_image and generated_image are your actual images
    #real_image = np.random.rand(28, 28)
    #generated_image = np.random.rand(28, 28)

    # Ensure images are in the range [0, 1]
    #real_image = np.clip(real_image, 0, 1)
    #generated_image = np.clip(generated_image, 0, 1)
    real_image = real_image.astype(np.float32)
    generated_image = generated_image.astype(np.float32)
    global high_class_max
    # Compute Structural Similarity Index (SSI)
    ssi_index, _ = ssim(real_image, generated_image, full=True, data_range=generated_image.max() - generated_image.min())
    #print(ssi_index)
    #print(type(ssi_index))
    #print(high_class_max)
    #time.sleep(20)
    if (ssi_index >= high_class_max):
        high_class_max = ssi_index
    #print(f"SSI Index: {ssi_index}")
    return ssi_index


    # Compute the Structural Similarity Index (SSI)
    #ssi_index, _ = ssim(real_image, generated_image, full=True)

    # The SSI value ranges from -1 to 1, with 1 being identical
    #print(f"SSI Index: {ssi_index}")

def plot_images(real_image, generated_image):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Plot real image
    axes[0].imshow(real_image, cmap='gray')
    axes[0].set_title('Real Image')
    axes[0].axis('off')

    # Plot generated image
    axes[1].imshow(generated_image, cmap='gray')
    axes[1].set_title('Generated Image')
    axes[1].axis('off')

    plt.show()


def sort_high_class_images(unique_images, c_train_images):

    high_class_images = []
    # Compare similarity numerically
    for i in range(0, len(unique_images)):
        print('Outer Loop: ', str(i))
        test_subject = unique_images[i].astype(int)
        print('Current Max: ', high_class_max)
        for j in range(0, len(c_train_images)):
            ssi_index = compare_structure(c_train_images[j], test_subject)
            #print(ssi_index)
            if ssi_index >= 0.89:
                #int_array = test_subject.astype(int)
                high_class_images.append(test_subject)
                break
            else:
                continue

    high_class_images_array = np.array(high_class_images)
    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Chest/high_class_generated_round2.npz', all_generated_images=high_class_images_array)
    print(len(high_class_images_array))


# Declare data sets
mnist_data = ['pneumoniamnist.npz', 'cardiomegaly.npz', 'normal_chest.npz', 'generated_images_train.npz', 'high_class_generated.npz']

# Get initial data sets
#choice = mnist_data[0]
#p_train_images, p_train_labels, p_val_images, p_val_labels, p_test_images, p_test_labels = prep_data(choice)

choice1 = mnist_data[1]
c_train_images, c_train_labels, c_val_images, c_val_labels, c_test_images, c_test_labels = prep_data(choice1)
print(len(c_train_images))

choice2 = mnist_data[4]
g_train_images, g_train_labels, g_val_images, g_val_labels, g_test_images, g_test_labels = prep_data(choice2)
print(len(g_train_images))

print(len(c_train_images))
print(len(g_train_images))

#unique_generated = np.unique(g_train_images)
#print(len(unique_generated))

flattened_images = g_train_images.reshape(g_train_images.shape[0], -1)
unique_indices = np.unique(flattened_images, axis=0, return_index=True)[1]

unique_images = g_train_images[unique_indices]
np.random.shuffle(unique_images)
print(len(unique_images))
#time.sleep(10)
print('c_train')
print(c_train_images[:5])
print('unique')
print(unique_images[:5])
#time.sleep(20)
real_image = c_train_images[0]
#generated_image = g_train_images[0]
generated_image = unique_images[0]


high_class_max = 0.89
#time.sleep(20)
sort_high_class_images(unique_images, c_train_images)
# Show them side by side
#plot_images(real_image, generated_image)