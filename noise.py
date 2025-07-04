import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def apply_gaussian_filter(image, sigma):
    return cv2.GaussianBlur(image, (5, 5), sigma)

def addition_noise(img_0, num_pixels):
    rw, cl = img_0.shape  # shape gives no of rows and columns
    # Salt and pepper noise
    for _ in range(num_pixels):
        # Salt section (white dots)
        y_crd = random.randint(0, rw-1)
        x_crd = random.randint(0, cl-1)
        img_0[y_crd, x_crd] = 255

    for _ in range(num_pixels):
        # Pepper section (black dots)
        y_crd = random.randint(0, rw-1)
        x_crd = random.randint(0, cl-1)
        img_0[y_crd, x_crd] = 0
       
    return img_0

def add_poisson_noise(image):
    noise = np.random.poisson(image.astype(float) / 255.0) * 255
    noisy_image = cv2.add(image, noise.astype(np.uint8))
    return noisy_image

def save_image(image, filename):
    cv2.imwrite(filename, image)

# Load an image from file
image_path = 'watermarked_certificate.png'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply Gaussian filter
sigma_values = [0.5, 1.0, 1.5]
filtered_images = [apply_gaussian_filter(original_image, sigma) for sigma in sigma_values]

# Save the filtered images
for i, sigma in enumerate(sigma_values):
    save_image(filtered_images[i], f'filtered_image_sigma_{sigma}.png')

# Add Poisson noise
poisson_noisy_image = add_poisson_noise(original_image)
cv2.imwrite("poison.png", poisson_noisy_image)

# Add Salt and Pepper noise
salt_pepper_noisy_image = addition_noise(original_image.copy(), 6000)
cv2.imwrite("Saltpepperlenna.png", salt_pepper_noisy_image)

# Display the original image
cv2.imshow('image-noise', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 