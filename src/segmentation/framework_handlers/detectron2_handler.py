from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json

from segments.utils import export_dataset
from src.annotation_handling.segmentsai_handler import SegmentsAIHandler
import os
import numpy as np

segmentsai_handler = SegmentsAIHandler()


def convert_detectron2_to_segments_format(image, outputs):
    segmentation_bitmap = np.zeros((image.shape[0], image.shape[1]), np.uint32)
    annotations = []
    counter = 1
    instances = outputs["instances"]
    for i in range(len(instances.pred_classes)):
        instance_id = counter
        mask = instances.pred_masks[i].cpu()
        segmentation_bitmap[mask] = instance_id
        trichome_annotation_category_id = (
            1  # assuming a single category for simplicity
        )
        annotations.append(
            {"id": instance_id, "category_id": trichome_annotation_category_id}
        )
        counter += 1
    return segmentation_bitmap, annotations

def convert_segments_to_detectron2_format(
    dataset_name, release_version, export_format="coco-instance"
):
    # GET THE DATASET INSTANCE WITH SEGMENTSAI_HANDLER
    dataset = segmentsai_handler.get_dataset_instance(dataset_name, version=release_version)
    # EXPORT THE DATASET
    export_file, image_dir = export_dataset(dataset, export_format=export_format)
    return dataset, export_file, image_dir


class Detectron2Handler:
    def __init__(
        self, config_path, model_weights_url, dataset_name, release_version, export_format="coco-instance", input_mask_format="bitmask", model_device="cuda"
    ):

        self.dataset, export_file, image_dir = convert_segments_to_detectron2_format(
            dataset_name, release_version, export_format
        )
        try:
            register_coco_instances(dataset_name, {}, export_file, image_dir)
        except Exception as e:
            print(f"Dataset registration failed: {e}")
        MetadataCatalog.get(dataset_name).set(
            thing_classes=[c.name for c in self.dataset.categories]
        )
        segments_metadata = MetadataCatalog.get(dataset_name)
        print(segments_metadata)
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.DATASETS.TRAIN = (dataset_name)
        self.cfg.DATASETS.TEST = ()
        self.cfg.INPUT.MASK_FORMAT = input_mask_format
        self.cfg.MODEL.WEIGHTS = model_weights_url
        self.cfg.MODEL.DEVICE = model_device
        self.predictor = DefaultPredictor(self.cfg)



    def setup_training(
        self,
        score_thresh_test=0.7,
        num_workers=2,
        ims_per_batch=2,
        base_lr=0.00025,
        max_iter=300,
        batch_size_per_image=128,
        num_classes=None,
    ):
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh_test
        self.cfg.DATALOADER.NUM_WORKERS = num_workers
        self.cfg.SOLVER.IMS_PER_BATCH = ims_per_batch
        self.cfg.SOLVER.BASE_LR = base_lr
        self.cfg.SOLVER.MAX_ITER = max_iter
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = batch_size_per_image
        # If num_classes is not provided, calculate it from dataset.categories
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = (
            num_classes if num_classes is not None else len(self.dataset.categories)
        )



    def train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(self.cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        
        self.cfg.MODEL.WEIGHTS = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        # self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = (
        # 0.7  # set the testing threshold for this model
        # )
        self.cfg.DATASETS.TEST = (dataset_name, )
        self.cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        


    def predict_image(self, image):
        outputs = self.predictor(image)
        segmentation_bitmap, annotations = convert_detectron2_to_segments_format(
            image, outputs
        )
        return segmentation_bitmap, annotations


if __name__ == "__main__":
    # Usage Example
    model_config_path = model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    model_weights = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    dataset_name = 'etaylor/cannabis_patches_all_images'  # Dataset name
    export_format = "coco-instance"  # Export format
    model_device = "cuda"  # Model device
    release_version = "v0.2"  # Dataset version

    handler = Detectron2Handler(model_config_path, model_weights, dataset_name, release_version)

    handler.setup_training()
    handler.train() 
 