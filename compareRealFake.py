
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
    counter = 0
    high_class_images_80 = []
    high_class_images_81 = []
    high_class_images_82 = []
    high_class_images_83 = []
    high_class_images_84 = []
    high_class_images_85 = []
    high_class_images_86 = []
    high_class_images_87 = []
    high_class_images_88 = []
    high_class_images_89 = []
    print('unique: ', len(unique_images))
    time.sleep(5)

    # Compare similarity numerically
    for i in range(0, len(unique_images)):
        print('Outer Loop: ', str(i))
        test_subject = unique_images[i].astype(int)
        print('Current Max: ', high_class_max)
        #print('Counter: ', counter)
        tracker = [0,0,0,0,0,0,0,0,0,0]
        for j in range(0, len(c_train_images)):
            ssi_index = compare_structure(c_train_images[j], test_subject)
            #print(ssi_index)
            if ssi_index >= 0.80:
                tracker[0] += 1#tracker[0] + 1
                if tracker[0] == 1:
                    high_class_images_80.append(test_subject)
            if ssi_index >= 0.81:
                tracker[1] += 1#1 tracker[1] + 1
                if tracker[1] == 1:
                    high_class_images_81.append(test_subject)
            if ssi_index >= 0.82:
                tracker[2] += 1#tracker[2] + 1
                if tracker[2] == 1:
                    high_class_images_82.append(test_subject)
            if ssi_index >= 0.83:
                tracker[3] += 1#tracker[3] + 1
                if tracker[3] == 1:
                    high_class_images_83.append(test_subject)
            if ssi_index >= 0.84:
                tracker[4] += 1#tracker[4] + 1
                if tracker[4] == 1:
                    high_class_images_84.append(test_subject)
            if ssi_index >= 0.85:
                tracker[5] += 1#tracker[5] + 1
                if tracker[5] == 1:
                    high_class_images_85.append(test_subject)
            if ssi_index >= 0.86:
                tracker[6] += 1#tracker[6] + 1
                if tracker[6] == 1:
                    high_class_images_86.append(test_subject)
            if ssi_index >= 0.87:
                tracker[7] += 1#tracker[7] + 1
                if tracker[7] == 1:
                    high_class_images_87.append(test_subject)
            if ssi_index >= 0.88:
                tracker[8] += 1#tracker[8] + 1
                if tracker[8] == 1:
                    high_class_images_88.append(test_subject)
            if ssi_index >= 0.89:
                tracker[9] += 1#tracker[9] + 1
                if tracker[9] == 1:
                    high_class_images_89.append(test_subject)
            
    """
    high_class_images_80 = list(set(high_class_images_80))
    high_class_images_81 = list(set(high_class_images_81))
    high_class_images_82 = list(set(high_class_images_82))
    high_class_images_83 = list(set(high_class_images_83))
    high_class_images_84 = list(set(high_class_images_84))
    high_class_images_85 = list(set(high_class_images_85))
    high_class_images_86 = list(set(high_class_images_86))
    high_class_images_87 = list(set(high_class_images_87))
    high_class_images_88 = list(set(high_class_images_88))
    high_class_images_89 = list(set(high_class_images_89))
    """
    high_class_images_80 = np.array(high_class_images_80)
    high_class_images_81 = np.array(high_class_images_81)
    high_class_images_82 = np.array(high_class_images_82)
    high_class_images_83 = np.array(high_class_images_83)
    high_class_images_84 = np.array(high_class_images_84)
    high_class_images_85 = np.array(high_class_images_85)
    high_class_images_86 = np.array(high_class_images_86)
    high_class_images_87 = np.array(high_class_images_87)
    high_class_images_88 = np.array(high_class_images_88)
    high_class_images_89 = np.array(high_class_images_89)
    """
    high_class_images_80 = np.unique(high_class_images_80)
    high_class_images_81 = np.unique(high_class_images_81)
    high_class_images_82 = np.unique(high_class_images_82)
    high_class_images_83 = np.unique(high_class_images_83)
    high_class_images_84 = np.unique(high_class_images_84)
    high_class_images_85 = np.unique(high_class_images_85)
    high_class_images_86 = np.unique(high_class_images_86)
    high_class_images_87 = np.unique(high_class_images_87)
    high_class_images_88 = np.unique(high_class_images_88)
    high_class_images_89 = np.unique(high_class_images_89)
    """
    #unique_array, index = np.unique(original_array, return_index=True)
    #high_class_images_array = np.array(high_class_images)
    #if len(high_class_images_80)

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_80_3.npz', all_generated_images=high_class_images_80)
    print('80: ', len(high_class_images_80))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_81_3.npz', all_generated_images=high_class_images_81)
    print('81: ', len(high_class_images_81))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_82_3.npz', all_generated_images=high_class_images_82)
    print('82: ', len(high_class_images_82))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_83_3.npz', all_generated_images=high_class_images_83)
    print('83: ', len(high_class_images_83))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_84_3.npz', all_generated_images=high_class_images_84)
    print('84: ', len(high_class_images_84))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_85_3.npz', all_generated_images=high_class_images_85)
    print('85: ', len(high_class_images_85))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_86_3.npz', all_generated_images=high_class_images_86)
    print('86: ', len(high_class_images_86))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_87_3.npz', all_generated_images=high_class_images_87)
    print('87: ', len(high_class_images_87))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_88_3.npz', all_generated_images=high_class_images_88)
    print('88: ', len(high_class_images_88))

    np.savez('/Users/digital_drifting/Desktop/CommandStation/Local/FileSystem/SchoolWork/FAUFall2023/DeepLearningGrad/medmnist_folder/Data/generated_train_89_3.npz', all_generated_images=high_class_images_89)
    print('89: ', len(high_class_images_89))


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
#g_train_images = g_train_images[:100]
#print(len(g_train_images))
#time.sleep(10)

print(len(c_train_images))
print(len(g_train_images))

#unique_generated = np.unique(g_train_images)
#print(len(unique_generated))

flattened_images = g_train_images.reshape(g_train_images.shape[0], -1)
unique_indices = np.unique(flattened_images, axis=0, return_index=True)[1]

unique_images = g_train_images[unique_indices]
np.random.shuffle(unique_images)
print(len(unique_images))
time.sleep(10)
print('c_train')
print(c_train_images[:5])
print('unique')
print(unique_images[:5])
#time.sleep(20)
real_image = c_train_images[0]
#generated_image = g_train_images[0]
generated_image = unique_images[0]


high_class_max = 0.80
#time.sleep(20)
sort_high_class_images(unique_images, c_train_images)
# Show them side by side
#plot_images(real_image, generated_image)