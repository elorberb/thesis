from src.annotation_handling.segmentsai_handler import SegmentsAIHandler
segmentsai_handler = SegmentsAIHandler()


if __name__ == "__main__":
    src_dataset_identifier = 'etaylor/cannabis_patches_week9_15_06_2023_3x_regular_IMG_2305'
    dest_dataset_identifier = "etaylor/all_cannabis_patches_multi_class"
    verbose = True

    # In[6]:
    segmentsai_handler.copy_dataset_contents(src_dataset_identifier, dest_dataset_identifier, verbose=verbose, only_patches=True)
