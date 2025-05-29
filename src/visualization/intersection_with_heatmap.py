import cv2


def intersection_with_heatmap(heatmap, mask):
    """
    Compute the intersection of the heatmap and the mask.
    """
    # Ensure heatmap and mask are in the same format
    if len(heatmap.shape) == 3:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)

    # Resize heatmap to match mask size
    heatmap_resized = cv2.resize(heatmap, (mask.shape[1], mask.shape[0]))

    # Threshold the heatmap to create a binary mask
    _, heatmap_binary = cv2.threshold(
        heatmap_resized, 0.5 * 255, 255, cv2.THRESH_BINARY
    )

    # Compute intersection
    intersection = cv2.bitwise_and(heatmap_binary, mask)

    return intersection
