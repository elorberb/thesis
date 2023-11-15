def detect_circles(path: str, total: int = None):
    """
    Detects circles in an image of trichomes, draws a red circle around each detected trichome, and displays the
    resulting image. Prints the total number of detected trichomes in the console.

    Args:
        path (str): The file path of the image to be processed.
        total (int, optional): The total number of trichomes in the image. If provided, this value will be
            included in the console output. Defaults to None.

    Returns:
        None

    """
    # Load the image
    img = cv2.imread(path)
    # Extract the image name from the file path
    name = path.split("/")[-1].split(".")[0]
    # Make a copy of the image
    img_copy = img.copy()
    # Convert the image to grayscale
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    # Convert the grayscale image to black and white
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Increase the threshold value to reduce noise
    thresh += 73
    black_white = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

    # Find contours in the black and white image
    cnts = cv2.findContours(black_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    count = 0
    # Loop over each contour
    for c in cnts:
        area = cv2.contourArea(c)
        _, _, w, h = cv2.boundingRect(c)
        ((x, y), r) = cv2.minEnclosingCircle(c)
        # If the contour has a small area, assume it is a trichome and draw a red circle around it
        if area < 20:  # note - this param can be learnable
            cv2.circle(img_copy, (int(x), int(y)), int(r), (0, 0, 255), -1)
            count += 1
    str_total = ""
    # If the total number of trichomes is provided, include it in the console output
    if total is not None:
        str_total = f", We labeled {total}"
    # Print the number of detected trichomes to the console
    print(f"\t\t\t\tName: {name}, Total Trichomes: {count}{str_total}")
    # Display the original image, black and white image, and image with red circles around trichomes side by side
    black_white = np.repeat(black_white[:, :, np.newaxis], 3, axis=2)
    horizontal = np.concatenate((img, black_white, img_copy), axis=1)
    cv2_imshow(horizontal)


def experiment_directory(directory):
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        detect_cirlces(f)
