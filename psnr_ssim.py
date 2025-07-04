from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import cv2

def psnr(original, reconstructed):
    # Assuming the images are in grayscale
    mse = np.mean((original - reconstructed) ** 2)
    max_pixel_value = 255.0
    psnr_value = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr_value

# Read the original and watermarked images
original_image = cv2.imread('cover.png', cv2.IMREAD_GRAYSCALE)
watermarked_image = cv2.imread('watermarked_certificate.png', cv2.IMREAD_GRAYSCALE)

# Check if images are loaded successfully
if original_image is None or watermarked_image is None:
    print("Error: Unable to read images.")
else:
    # Ensure images have the same size
    if original_image.shape == watermarked_image.shape:
        # Calculate SSIM
        ssim_score = compare_ssim(original_image, watermarked_image)
        print("SSIM Value:", ssim_score)

        # Calculate PSNR
        psnr_value = psnr(original_image, watermarked_image)
        print(f"PSNR Value: {psnr_value} dB")
    else:
        print("Error: Images have different dimensions.")