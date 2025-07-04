import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, image_size=(256, 256), font_size=20, bg_color=(0,0,0), text_color=(255,255,255)):
    """
    Convert text to an image with centered text
    
    Parameters:
    - text: Text to be converted
    - image_size: Size of output image (width, height)
    - font_size: Size of font
    - bg_color: Background color (RGB)
    - text_color: Text color (RGB)
    
    Returns:
    - NumPy array of the text image
    """
    image = Image.new("RGB", image_size, bg_color)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
    
    # Get text bounding box
    text_bbox = draw.textbbox((0, 0), text, font=font)
    
    # Calculate text width and height
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Center text
    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    
    draw.text((x, y), text, font=font, fill=text_color)
    return np.array(image)

def arnold_scramble(image, iterations=1):
    """
    Apply Arnold's cat map scrambling to an image
    
    Parameters:
    - image: Input image (NumPy array)
    - iterations: Number of scrambling iterations
    
    Returns:
    - Scrambled image
    """
    height, width = image.shape[:2]
    scrambled = image.copy()
    
    for _ in range(iterations):
        temp = scrambled.copy()
        for x in range(width):
            for y in range(height):
                new_x = (x + y) % width
                new_y = (x + 2*y) % height
                scrambled[new_y, new_x] = temp[y, x]
    
    return scrambled

def arnold_unscramble(image, iterations=1):
    """
    Reverse Arnold's cat map scrambling
    
    Parameters:
    - image: Scrambled image
    - iterations: Number of unscrambling iterations (should match scrambling iterations)
    
    Returns:
    - Unscrambled image
    """
    height, width = image.shape[:2]
    unscrambled = image.copy()
    
    for _ in range(iterations):
        temp = unscrambled.copy()
        for x in range(width):
            for y in range(height):
                new_x = (2*x - y) % width
                new_y = (-x + y) % height
                unscrambled[new_y, new_x] = temp[y, x]
    
    return unscrambled

def normalize_image(image):
    """
    Normalize image to range [0, 255]
    
    Parameters:
    - image: Input image
    
    Returns:
    - Normalized image
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)