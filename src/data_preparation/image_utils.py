import cv2

def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height))
    return resized_image
