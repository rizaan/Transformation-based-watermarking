import cv2
import numpy as np
import pywt
import pytesseract
from text_watermark_utils import arnold_unscramble, normalize_image

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_watermark(watermarked_image_path, scramble_iterations=1):
    """
    Extract watermark from a DWT-DCT watermarked image
   
    Parameters:
    - watermarked_image_path: Path to watermarked image
    - scramble_iterations: Arnold unscrambling iterations
   
    Returns:
    - Extracted watermark text
    """
    # Read watermarked image
    host = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    watermarked_image = cv2.resize(host, (1024, 1024))
   
    # Apply DWT
    wavelet = 'haar'
    coeffs_w = pywt.dwt2(watermarked_image, wavelet)
    LL_w, (LH_w, HL_w, HH_w) = coeffs_w
    LL2_w, (LH2_w, HL2_w, HH2_w) = pywt.dwt2(HH_w, wavelet)
   
    # Extract watermark from HH2 subband
    rows, cols = HH2_w.shape
    extracted_dct = np.zeros_like(HH2_w, dtype=np.float32)
   
    # Copy watermark coefficients
    for i in range(rows):
        for j in range(0, cols-i):
            extracted_dct[i, j] = HH2_w[i, j]
   
    # Normalize and apply inverse DCT
    extracted_watermark = cv2.idct(extracted_dct)
   
    # Unscramble watermark
    unscrambled_watermark = arnold_unscramble(extracted_watermark, iterations=scramble_iterations)
   
    # Save extracted watermark
    extracted_watermark_normalized = normalize_image(unscrambled_watermark)
    cv2.imwrite("extracted_watermark.png", extracted_watermark_normalized)
   
    # Extract text using Tesseract
    extracted_text = pytesseract.image_to_string('extracted_watermark.png', lang='eng', config='--psm 6')
   
    return extracted_text.strip()

# Example usage
if __name__ == "__main__":
    watermarked_image_path = "watermarked_certificate.png"
    extracted_text = extract_watermark(watermarked_image_path)
    print("Extracted Text:", extracted_text)