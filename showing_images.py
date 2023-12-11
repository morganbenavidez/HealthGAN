import matplotlib.pyplot as plt

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

def plot_images(real_image, generated_image, denormalized_tanh, denormalized_sigmoid):
    fig, axes = plt.subplots(1, 4, figsize=(8, 4))

    # Plot real image
    axes[0].imshow(real_image, cmap='gray')
    axes[0].set_title('Real Image')
    axes[0].axis('off')

    # Plot generated image
    axes[1].imshow(generated_image, cmap='gray')
    axes[1].set_title('Generated Image')
    axes[1].axis('off')

    # Plot generated image
    axes[2].imshow(denormalized_tanh, cmap='gray')
    axes[2].set_title('Denormalized Image')
    axes[2].axis('off')

    # Plot generated image
    axes[3].imshow(denormalized_sigmoid, cmap='gray')
    axes[3].set_title('Denormalized Image')
    axes[3].axis('off')

    plt.show()

plot_images(train_images[0], normalized_train_images[0], denormalized_tanh[0], denormalized_pixels[0])