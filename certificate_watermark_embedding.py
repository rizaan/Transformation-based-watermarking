import cv2
import numpy as np
import pywt
from text_watermark_utils import text_to_image, arnold_scramble

def embed_watermark(host_image_path, watermark_text, watermark_iterations=1, watermark_image_path="watermark_image.png"):
    """
    Embed a text watermark into a host image using DCT and DWT
    
    Parameters:
    - host_image_path: Path to the host image
    - watermark_text: Text to embed as watermark
    - watermark_iterations: Arnold scrambling iterations
    - watermark_image_path: Path to save the watermark image
    
    Returns:
    - Watermarked image
    """
    # Read host image
    host = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
    original_height, original_width = host.shape
    
    # Resize host image
    host_image = cv2.resize(host, (1024, 1024))
    
    # Create watermark image
    watermark_image = text_to_image(watermark_text, image_size=(256, 256), font_size=20)
    watermark_image_gray = cv2.cvtColor(watermark_image, cv2.COLOR_BGR2GRAY)
    
    # Save the watermark image
    cv2.imwrite(watermark_image_path, watermark_image_gray)
    
    # Apply Arnold scrambling to watermark
    scrambled_watermark = arnold_scramble(watermark_image_gray, iterations=watermark_iterations)
    
    # Convert watermark to float32
    watermark_float32 = np.float32(scrambled_watermark)
    
    # Apply DCT to watermark
    watermark_dct = cv2.dct(watermark_float32)
    
    # Modify DCT coefficients
    rows, cols = watermark_dct.shape
    for i in range(rows):
        for j in range(cols - i, cols):
            watermark_dct[i, j] = 0
    
    # Apply DWT to host image
    wavelet = 'haar'
    coeffs = pywt.dwt2(host_image, wavelet)
    LL, (LH, HL, HH) = coeffs
    LL2, (LH2, HL2, HH2) = pywt.dwt2(HH, wavelet)
    
    # Embed watermark in HH2 subband
    for i in range(rows):
        for j in range(0, cols-i):
            HH2[i, j] = watermark_dct[i, j]
    
    # Reconstruct image
    coeffs_2 = (LL2, (LH2, HL2, HH2))
    watermarked_hh = pywt.idwt2(coeffs_2, wavelet)
    watermarked_image = pywt.idwt2((LL, (LH, HL, watermarked_hh)), wavelet)
    
    # Normalize and save
    watermarked_image_normalized = watermarked_image / 255
    cv2.imwrite("watermarked_certificate.png", 
                cv2.resize(watermarked_image_normalized * 255, (original_width, original_height)))
    
    return watermarked_image_normalized

# Example usage
if __name__ == "__main__":
    watermark_text = str(input("Enter Text [Limit upto 5 Characters length]:"))
    host_image_path = "cover.png"
    watermark_image_path = "watermark_image.png"
    embed_watermark(host_image_path, watermark_text, watermark_image_path=watermark_image_path)