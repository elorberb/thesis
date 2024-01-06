import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image


def show_anns(anns):
    # If there are no annotations, exit the function early
    if len(anns) == 0:
        return

    # Sort the annotations by area in descending order
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    # Get the current axis (plot) on which to draw
    ax = plt.gca()
    # Disable autoscaling of the axis
    ax.set_autoscale_on(False)

    # Create a transparent image of the same size as the first (largest) annotation
    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0

    # Loop over each annotation
    for ann in sorted_anns:
        # Get the mask for this annotation
        m = ann["segmentation"]

        # Generate a random color for this mask (RGB + alpha, where alpha is 0.35)
        color_mask = np.concatenate([np.random.random(3), [0.35]])

        # Apply the color to the mask on the image
        img[m] = color_mask

    # Display the image with the colored masks
    ax.imshow(img)


def plot_masks(y_true, y_pred):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    image = y_true["image"]
    y_true_mask = y_true["instance_bitmap"]
    y_pred_mask = y_pred["mask"]

    # Plot Original image
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Plot ground truth
    ax[1].imshow(image)
    ax[1].imshow(
        y_true_mask, alpha=0.5, cmap="jet"
    )  # Overlays the mask on the image, adjust alpha to your needs
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    # Plot prediction
    ax[2].imshow(image)
    show_anns(y_pred_mask)
    ax[2].set_title("Prediction")
    ax[2].axis("off")

    plt.show()


def plot_confusion_matrix(TPs, FPs, FNs):
    confusion_matrix = [
        [TPs, FPs],
        [FNs, 0],
    ]  # Note that we don't have TNs in this case

    # Create a heatmap
    plt.figure(figsize=(6, 4))
    heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

    # Set labels
    heatmap.xaxis.set_ticklabels(["Positive", "Negative"])
    heatmap.yaxis.set_ticklabels(["Positive", "Negative"])
    heatmap.xaxis.tick_top()  # x axis on top
    heatmap.xaxis.set_label_position("top")

    # Add axis labels
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.show()


def plot_iou_histogram(iou_values, bins=10):
    plt.figure(figsize=(10, 6))
    plt.hist(iou_values, bins=bins, edgecolor="black")
    plt.xlabel("IOU")
    plt.ylabel("Frequency")
    plt.title("Histogram of IOU Values")
    plt.show()


def plot_segments_one_by_one(image, masks):
    # Sort the masks by area in descending order
    sorted_masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)

    # Iterate over all masks
    for i, mask in enumerate(sorted_masks, start=1):
        # Extract bounding box coordinates
        x, y, width, height = mask["bbox"]

        # Use the bounding box coordinates to extract the segment from the image
        segment = image[y : y + height, x : x + width]

        # Plot the segment
        plt.figure(figsize=(5, 5))
        plt.imshow(segment)
        plt.axis("off")
        plt.title(f"Segment {i}")
        plt.show()


def show_anns_numbered(anns):
    # If there are no annotations, exit the function early
    if len(anns) == 0:
        return

    # Sort the annotations by area in descending order
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    # Get the current axis (plot) on which to draw
    ax = plt.gca()
    # Disable autoscaling of the axis
    ax.set_autoscale_on(False)

    # Create a transparent image of the same size as the first (largest) annotation
    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0

    # Loop over each annotation
    for i, ann in enumerate(sorted_anns):
        # Get the mask for this annotation
        m = ann["segmentation"]

        # Generate a random color for this mask (RGB + alpha, where alpha is 0.35)
        color_mask = np.concatenate([np.random.random(3), [0.35]])

        # Apply the color to the mask on the image
        img[m] = color_mask

        # Calculate the centroid of the mask
        y, x = np.argwhere(m).mean(axis=0)

        # Add a text label with the segment number at the centroid of the mask
        plt.text(x, y, str(i + 1), color="white", fontsize=12, ha="center", va="center")

    # Display the image with the colored masks
    ax.imshow(img)


def plot_all_segments(image, masks):
    # Create a figure and axes
    fig, ax = plt.subplots(1, figsize=(10, 10))

    # Display the image
    ax.imshow(image)

    # Call the function to overlay the masks
    show_anns_numbered(masks)

    # Hide the axes
    ax.axis("off")

    plt.show()


def plot_images(directory_path):
    # Collecting all the file names in the directory
    image_files = os.listdir(directory_path)

    # Filtering the list to only include JPEG and PNG images
    image_files = [f for f in image_files if f.endswith((".jpg", ".jpeg", ".png"))]

    # The number of images
    n_images = len(image_files)

    # Create subplots: adjust the size and the number of columns as per your requirement
    ncols = 4
    nrows = n_images // ncols + (n_images % ncols > 0)

    # Create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows))

    for i, image_file in enumerate(image_files):
        img = Image.open(os.path.join(directory_path, image_file))
        ax[i // ncols, i % ncols].imshow(img)
        ax[i // ncols, i % ncols].set_title(image_file)
        ax[i // ncols, i % ncols].axis("off")

    # Remove empty subplots
    for j in range(i + 1, nrows * ncols):
        fig.delaxes(ax.flatten()[j])

    plt.tight_layout()
    plt.show()


# ---- Bounding Box Ploting Functions ----

import matplotlib.patches as patches


def plot_cut_images(cut_images):
    num_images = len(cut_images)

    if num_images == 0:
        print("No cut images to plot.")
        return

    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))

    # If there's only one image, axes will be an 'Axes' object and not an array.
    # Convert it to an array for consistency.
    if num_images == 1:
        axes = np.array([axes])

    for ax, (label, cut_image) in zip(axes, cut_images.items()):
        # Convert the image from BGR to RGB
        rgb_image = cv2.cvtColor(cut_image, cv2.COLOR_BGR2RGB)

        ax.imshow(rgb_image)
        ax.set_title(label)
        ax.axis("off")

    plt.show()


def plot_with_bboxes(image, bboxes, classification_results=None):
    # Convert the image from BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, figsize=(10, 10))
    # ax.imshow(rgb_image)

    for label, (y_min, y_max, x_min, x_max) in bboxes.items():
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect)

        # Display classification result
        if classification_results and label in classification_results:
            ax.text(
                x_min,
                y_min,
                classification_results[label],
                color="blue",
                fontweight="bold",
                fontsize=12,
                ha="left",
                va="bottom",
            )

    ax.axis("off")
    plt.show()
