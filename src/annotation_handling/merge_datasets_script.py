from src.annotation_handling.segmentsai_handler import SegmentsAIHandler
segmentsai_handler = SegmentsAIHandler()
import config


if __name__ == "__main__":
    # src_dataset_identifier = 'etaylor/cannabis_patches_week9_15_06_2023_3x_regular_IMG_2198'
    dest_dataset_identifier = "etaylor/cannabis_patches_all_11_4_24"
    verbose = True
    
    # merge train and test datasets to all images dataset created in 11.4.24
    segmentsai_handler.copy_dataset_contents(source_dataset_id="etaylor/cannabis_patches_train",
                                             destination_dataset_id=dest_dataset_identifier,
                                             verbose=verbose,
                                             only_patches=True)
    segmentsai_handler.copy_dataset_contents(source_dataset_id="etaylor/cannabis_patches_test",
                                             destination_dataset_id=dest_dataset_identifier,
                                             verbose=verbose,
                                             only_patches=True)
    # # merge datasets
    # segmentsai_handler.copy_dataset_contents("etaylor/cannabis_patches_week9_15_06_2023_3x_regular_IMG_2242", dest_dataset_identifier, verbose=verbose, only_patches=True)
    # segmentsai_handler.copy_dataset_contents("etaylor/cannabis_patches_week9_15_06_2023_3x_regular_IMG_2305", dest_dataset_identifier, verbose=verbose, only_patches=True)
    # segmentsai_handler.copy_dataset_contents("etaylor/cannabis_patches_week9_15_06_2023_3x_regular_IMG_2285", dest_dataset_identifier, verbose=verbose, only_patches=True)
    
    # decrease label category ids - some of the datasets catagories are not in order (start with 1 and not 0)
    # segmentsai_handler.decrement_label_category_ids("etaylor/cannabis_patches_week9_15_06_2023_3x_regular_IMG_2242")
    # segmentsai_handler.decrement_label_category_ids("etaylor/cannabis_patches_week9_15_06_2023_3x_regular_IMG_2305")
