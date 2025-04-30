import os
import cv2
from tqdm import tqdm  # Optional for progress bar

def process_image_pairs(original_dir, modified_dir, output_dir, img_extensions=('.jpg', '.png', '.jpeg'), pixel_size=10, method='default', colored=False):
    """
    Process pairs of images from two directories and save masks to a third directory.
    
    Args:
        original_dir: Path to directory with original images
        modified_dir: Path to directory with modified images
        output_dir: Path to save generated masks
        img_extensions: Tuple of valid image extensions to process
        pixel_size: Size of the pixel for pixelised mask computation
        method: Method to compute the mask ('default' or 'removed_lines')
        colored: Boolean to indicate if colored mask should be generated
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files in original directory
    original_files = [f for f in os.listdir(original_dir) 
                     if f.lower().endswith(img_extensions)]
    
    # Process each image pair
    for filename in tqdm(original_files, desc="Processing images"):
        original_path = os.path.join(original_dir, filename)
        modified_path = os.path.join(modified_dir, filename)
        
        # Skip if modified image doesn't exist
        if not os.path.exists(modified_path):
            continue
            
        try:
            # Read images using OpenCV
            original = cv2.imread(original_path)
            modified = cv2.imread(modified_path)
            
            if original is None or modified is None:
                print(f"Could not read image pair: {filename}")
                continue
                
            # Compute mask
            if method == 'default':
                if pixel_size == 1:
                    mask = get_mask(original, modified)
                else:
                    mask = get_mask_pixelised(original, modified, pixel_size=pixel_size)
            elif method == 'removed_lines':
                mask = get_removed_lines(original, modified, colored=colored)
            else:
                raise ValueError("Invalid method specified. Use 'default' or 'removed_lines'.")
            
            # Save mask preserving original filename
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, mask)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def get_mask(original_image, modified_image):
    """
    Compute the mask by subtracting the modified image from the original image.
    """
    mask = cv2.absdiff(original_image, modified_image)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Binarize the mask
    return mask

# Usage

def get_mask_pixelised(original_image, modified_image, pixel_size=10):
    """
    Compute the mask by subtracting the modified image from the original image.
    """
    import numpy as np
    mask = cv2.absdiff(original_image, modified_image)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Binarize the mask

    # for each macro pixel, if one of the pixels is 1, set the macro pixel to 1
    macro_mask = np.zeros((mask.shape[0] // pixel_size, mask.shape[1] // pixel_size), dtype=np.uint8)
    for i in range(macro_mask.shape[0]):
        for j in range(macro_mask.shape[1]):
            macro_mask[i, j] = np.max(mask[i*pixel_size:(i+1)*pixel_size, j*pixel_size:(j+1)*pixel_size])
    return macro_mask

def get_removed_lines(original_image, modified_image, colored=True):
   # Let's adjust the threshold to ensure all lines (including blue ones) are captured.
    # Specifically, let's widen the brightness (V) threshold to capture all colors more accurately.
    import numpy as np

    # Convert to HSV again
    hsv_original = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    hsv_modified = cv2.cvtColor(modified_image, cv2.COLOR_BGR2HSV)

    # Adjusted mask: capture all non-white pixels more aggressively
    lower_color = np.array([0, 30, 30])  # increase lower saturation and value to avoid near-white
    upper_color = np.array([180, 255, 255])
    mask_original = cv2.inRange(hsv_original, lower_color, upper_color)
    mask_modified = cv2.inRange(hsv_modified, lower_color, upper_color)

    # Subtract modified mask from original mask
    diff_mask = cv2.subtract(mask_original, mask_modified)

    # Clean up small noise
    kernel = np.ones((3,3), np.uint8)
    diff_mask_clean = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to get colored differences
    colored_diff = cv2.bitwise_and(original_image, original_image, mask=diff_mask_clean)

    # Convert for matplotlib display (BGR to RGB)
    # colored_diff_rgb = cv2.cvtColor(colored_diff, cv2.COLOR_BGR2RGB)

    if colored:
        return colored_diff
    else:
        return diff_mask_clean
    

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2:
        numeros = [sys.argv[1]]
    else:
        numeros = range(2, 8)
    for numero in numeros:
        # Call the function with the specified directories
        process_image_pairs(
            original_dir="MSH/MSH/plots/configuration1",
            modified_dir=f"MSH/MSH/plots/configuration{numero}",
            output_dir=f"data/MSH/mask_removed_color/mask{numero}",
            pixel_size=1,
            method="removed_lines",
            colored=True

        )
        process_image_pairs(
            original_dir="MSH/MSH/plots/configuration1",
            modified_dir=f"MSH/MSH/plots/configuration{numero}",
            output_dir=f"data/MSH/mask_removed/mask{numero}",
            pixel_size=1,
            method="removed_lines",
            colored=False
        )

        process_image_pairs(
            original_dir="MSH/MSH/plots/configuration1",
            modified_dir=f"MSH/MSH/plots/configuration{numero}",
            output_dir=f"data/MSH/mask_classic/mask{numero}",
            pixel_size=1,
            method="default",
            colored=True
        )    