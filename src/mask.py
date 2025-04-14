import os
import cv2
from tqdm import tqdm  # Optional for progress bar

def process_image_pairs(original_dir, modified_dir, output_dir, img_extensions=('.jpg', '.png', '.jpeg')):
    """
    Process pairs of images from two directories and save masks to a third directory.
    
    Args:
        original_dir: Path to directory with original images
        modified_dir: Path to directory with modified images
        output_dir: Path to save generated masks
        img_extensions: Tuple of valid image extensions to process
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
            mask = get_mask(original, modified)
            
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

if __name__ == "__main__":
    process_image_pairs(
        original_dir="MSH/MSH/plots/configuration1",
        modified_dir="MSH/MSH/plots/configuration7",
        output_dir="data/MSH/mask"
    )