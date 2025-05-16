from torchvision import transforms


def get_transform(image_size=(284, 284)):
    """Return the transform to resize the image to the specified size.
    Andditionally, convert it to a tensor.
    """
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),  # Convert numpy array (from cv2) to PIL Image
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )
    return transform
