import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import sys



def setup_sam():
    !{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam)
    return sam, mask_generator


def segments_patches_SAM(images_names, mask_generator):
    segmentation_dict = {}

    for image, name in images_names:
        mask = mask_generator.generate(image)
        num_segments = len(mask)

        # Initialize an array of False with the same shape as the segment masks
        instance_bitmap = np.zeros_like(mask[0]['segmentation'], dtype=bool)

        # create a single instance bitmap
        for seg in mask:
            instance_bitmap = np.logical_or(instance_bitmap, seg['segmentation'])

        segmentation_dict[name] = {
            'mask': mask,
            'num_segments': num_segments,
            "instance_bitmap": instance_bitmap,
        }

    return segmentation_dict
