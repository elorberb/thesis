def detect_circles(img, minRadius, maxRadius, param1=118, param2=8, save_path=None):
    """
    detect circles in an image using HoughCircles algorithm.
    :param img:
    :param minRadius:
    :param maxRadius:
    :param param1:
    :param param2:
    :param save_path:
    :return:
    """
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    circles = cv2.HoughCircles(
        img,
        cv2.HOUGH_GRADIENT,
        1,
        5,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius,
    )

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 1)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 1)

    cv2.imwrite(save_path, cimg)  # save the image to the specified path
    cv2_imshow(cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(f"number of circles detected {circles.shape[1]}")
