from fastai.vision.all import *
from fastai.vision import *
import PIL

# models to compare
image_classification_models = {
    "alexnet": models.alexnet,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50_32x4d": models.resnext50_32x4d,
    "resnext101_32x8d": models.resnext101_32x8d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2,
    "vgg16_bn": models.vgg16_bn,
    "vgg19_bn": models.vgg19_bn,
    "squeezenet1_1": models.squeezenet1_1,
    "densenet121": models.densenet121,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
}

small_available_models = {
    "alexnet": models.alexnet,
    "resnet34": models.resnet34,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
}


def load_classification_model(model_path):
    """Load your model from the specified path"""
    model = load_learner(model_path)
    return model


# Custom Transform to Resize with Padding
def custom_transform(size):
    return Resize(size, method="pad", pad_mode="zeros")


# Custom Transform to HSV using fastai's rgb2hsv
class RGB2HSV(Transform):
    def encodes(self, img: PILImage):
        return rgb2hsv(img)


def get_dataloaders(dataset_path):
    # Define all DataLoaders configurations
    dataloaders_dict = {
        "raw": ImageDataLoaders.from_folder(
            path=dataset_path, item_tfms=Resize(128), bs=16, valid_pct=0.25
        ),
        "resize_with_padding": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=custom_transform(128),  # Adjust size as needed
            batch_tfms=aug_transforms(size=128),  # Apply data augmentation
            bs=16,
            valid_pct=0.25,
        ),
        "convert_to_hsv": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=Resize(128),  # Resize before applying HSV transform
            batch_tfms=[RGB2HSV(), *aug_transforms(size=128)],
            bs=16,
            valid_pct=0.25,
        ),
        "normalize_pixels": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=Resize(128),  # Resize before applying HSV transform
            batch_tfms=[
                Normalize.from_stats(*imagenet_stats),
                *aug_transforms(size=128),
            ],
            bs=16,
            valid_pct=0.25,
        ),
        "brightness_contrast": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=Resize(128),
            batch_tfms=[
                Brightness(max_lighting=0.2, p=0.75),
                Contrast(max_lighting=0.2, p=0.75),
                *aug_transforms(size=128),
            ],
            bs=16,
            valid_pct=0.25,
        ),
        "combined_resize_padding_hsv": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=custom_transform(size=128),  # Resize and HSV transform
            batch_tfms=[
                RGB2HSV(),
                *aug_transforms(size=128, flip_vert=True, max_rotate=10),
            ],
            bs=16,
            valid_pct=0.25,
        ),
        "combined_hsv_normalize": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=Resize(128),  # Resize before applying HSV transform
            batch_tfms=[
                RGB2HSV(),
                Normalize.from_stats(*imagenet_stats),
                *aug_transforms(size=128),
            ],
            bs=16,
            valid_pct=0.25,
        ),
        "combined_padding_normalize": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=custom_transform(128),  # Resize with padding
            batch_tfms=[
                Normalize.from_stats(*imagenet_stats),
                *aug_transforms(size=128),
            ],
            bs=16,
            valid_pct=0.25,
        ),
        "combined_padding_brightness_contrast": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=custom_transform(128),  # Resize with padding
            batch_tfms=[
                Brightness(max_lighting=0.2, p=0.75),
                Contrast(max_lighting=0.2, p=0.75),
                *aug_transforms(size=128),
            ],
            bs=16,
            valid_pct=0.25,
        ),
        "combined_hsv_brightness_contrast": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=Resize(128),  # Resize before applying HSV transform
            batch_tfms=[
                RGB2HSV(),
                Brightness(max_lighting=0.2, p=0.75),
                Contrast(max_lighting=0.2, p=0.75),
                *aug_transforms(size=128),
            ],
            bs=16,
            valid_pct=0.25,
        ),
        "combined_all": ImageDataLoaders.from_folder(
            path=dataset_path,
            item_tfms=custom_transform(size=128),  # Resize and HSV transform
            batch_tfms=[
                RGB2HSV(),
                *aug_transforms(size=128, flip_vert=True, max_rotate=10),
                Brightness(max_lighting=0.2, p=0.75),
                Contrast(max_lighting=0.2, p=0.75),
            ],
            bs=16,
            valid_pct=0.25,
        ),
    }

    return dataloaders_dict
