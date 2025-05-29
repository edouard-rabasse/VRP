import cv2


def get_mask(original_image, modified_image):
    """
    Compute the mask by subtracting the modified image from the original image.
    """
    mask = cv2.absdiff(original_image, modified_image)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)  # Binarize the mask
    return mask
