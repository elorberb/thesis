from segments import SegmentsClient, SegmentsDataset
from segments.utils import bitmap2file

import numpy as np
import os
import matplotlib.pyplot as plt

from dotenv import load_dotenv
load_dotenv()


class SegmentsAIHandler:
    def __init__(self):
        api_key = os.getenv("SEGMENTS_API_KEY")
        self.client = SegmentsClient(api_key)
        
        

    def get_dataset_instance(self, dataset_name, version='v0.1', labelset='ground-truth', filter_by=None, output_dir="segments"):
        """
        Obtain a SegmentsDataset instance.

        Parameters:
        - dataset_name (str): The name of the dataset you want to retrieve.
        - version (str): The version of the dataset release.
        - labelset (str): The labelset to use, default is 'ground-truth'.
        - filter_by (list): A list of filters to apply (e.g., ['LABELED', 'REVIEWED']).

        Returns:
        - SegmentsDataset: An instance of the SegmentsDataset.
        """
        # Get the release file for the specified dataset and version
        release = self.client.get_release(dataset_name, version)
        output_dir = os.path.join(output_dir, "segments")
        # Initialize and return the SegmentsDataset
        return SegmentsDataset(release, segments_dir=output_dir)

        
        
    def create_new_dataset(self, dataset_identifier, dataset_description, task_type, task_attributes):
        """
        Create a new dataset using the Segments AI API.

        Example:
            Dataset details for a generic dataset:

            dataset_identifier = "trichomes_patches_week8_3x_regular_IMG_1802"
            dataset_description = "Trichomes patches of image IMG_1802 - purple flower."
            task_type = "segmentation-bitmap"
            task_attributes = {
                "format_version": "0.1",
                "categories": [
                    {"name": "trichome", "id": 1},
                ]
            }

            #Create the dataset:

            dataset = create_dataset(name, dataset_description, task_type, task_attributes)
            print(dataset)
        """
        new_dataset = self.client.add_dataset(dataset_identifier, dataset_description, task_type, task_attributes)
        return new_dataset     
    
    
    def add_collaborator_to_dataset(self, dataset_id, user, user_role='annotator'):
        """
        Add a collaborator to a dataset on Segments.ai.

        Parameters:
        - dataset_id: Identifier for the dataset (format: 'owner_name/dataset_name')
        - user: The username of the collaborator to be added
        - user_role: The role assigned to the collaborator ('viewer', 'annotator', 'reviewer', or 'admin')

        Returns:
        - Response from the Segments.ai API after attempting to add the collaborator
        """
        return self.client.add_dataset_collaborator(dataset_id, user, user_role)
    
    
    def upload_images(self, dataset_identifier, images_folder_path):
        """
        Uploads image files from a specified directory to a Segments.ai dataset.

        Parameters:
        - dataset_identifier: The name of the dataset on Segments.ai.
        - images_folder_path: Local system path to the folder containing images.
        - image_extension: File extension of images to upload (default is .jpg).
        """
        # Retrieve a list of image filenames with the given file extension
        image_filenames = [filename for filename in os.listdir(images_folder_path)]

        # Upload each image to the Segments.ai dataset
        for filename in image_filenames:
            file_path = os.path.join(images_folder_path, filename)  # Full path to image

            # Open the image file and upload it as an asset to Segments.ai
            with open(file_path, "rb") as image_file:
                uploaded_asset = self.client.upload_asset(image_file, filename)

            # Get the URL of the uploaded asset
            uploaded_image_url = uploaded_asset.url

            # Define the sample attributes with the uploaded image URL
            sample_attributes = {"image": {"url": uploaded_image_url}}

            # Add the new sample to the dataset on Segments.ai
            new_sample = self.client.add_sample(dataset_identifier, filename, sample_attributes)
            print(f"Uploaded {filename} and added as sample: {new_sample}")
            
    
    def visualize_sample(self, *args):
        """
        Visualizes a single sample with its segmentation bitmap overlay.   
        """
        images = args
        for i, image in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(np.array(image))
        plt.show()

    
    def visualize_dataset(self, dataset_identifier, dataset_release_version="v0.1"):
        """
        Visualize samples in a Segments.ai dataset.
        """
        # Fetch the specified dataset release
        dataset_release = self.client.get_release(dataset_identifier, dataset_release_version)
        dataset_samples = SegmentsDataset(dataset_release)

        # Iterate and visualize each sample in the dataset
        for sample in dataset_samples:
            try:
                self.visualize_sample(sample["image"], sample["segmentation_bitmap"])
            except TypeError as error:
                print(f"Could not visualize sample {sample['name']}: {error}")
                
                
    def upload_annotation_for_sample(self, sample_uuid, segmentation_bitmap, annotation_data):
        """
        Uploads annotation data for a specific sample to Segments.ai.

        Parameters:
        - sample_uuid: The UUID of the sample to which the annotation will be uploaded.
        - segmentation_bitmap: The segmentation bitmap data from the model's prediction.
        - annotation_data: The corresponding annotation data for the segmentation.
        """

        # Convert the segmentation bitmap to a file-like object
        segmentation_file = bitmap2file(segmentation_bitmap)

        # Upload the segmentation bitmap as an asset
        uploaded_asset = self.client.upload_asset(segmentation_file, "label.png")
        uploaded_asset_url = uploaded_asset.url

        # Prepare the attributes for the uploaded label
        label_attributes = {
            "format_version": "0.1",
            "annotations": annotation_data,
            "segmentation_bitmap": {"url": uploaded_asset_url},
        }

        # Add the label to the specific sample on Segments.ai
        self.client.add_label(
            sample_uuid, "ground-truth", label_attributes, label_status="PRELABELED"
        )
        
    
    def copy_dataset_contents(self, source_dataset_id, destination_dataset_id, verbose=False):
        """
        Copies all samples and their annotations from one dataset to another.

        Parameters:
        - source_dataset_id: Identifier for the source dataset.
        - destination_dataset_id: Identifier for the destination dataset.
        - verbose: Flag to enable verbose logging of the process.
        """
        source_samples = self.client.get_samples(source_dataset_id)

        for index, sample in enumerate(source_samples):
            self.copy_sample(sample, destination_dataset_id, index, verbose)



    def copy_sample(self, sample, destination_dataset_id, index, verbose):
        # Copy an individual sample and its annotations to another dataset
        if verbose:
            print(f"Processing sample {index + 1}: {sample.name}")

        label = self.client.get_label(sample.uuid, labelset="ground-truth")
        if label:
            self.copy_sample_and_label(sample, label, destination_dataset_id, verbose)


    def copy_sample_and_label(self, sample, label, destination_dataset_id, verbose):
        # Add the sample and its label to the destination dataset
        new_sample = self.client.add_sample(
            destination_dataset_id, sample.name, sample.attributes
        )
        self.client.add_label(new_sample.uuid, "ground-truth", label.attributes)

        if verbose:
            if new_sample:
                print(f"  - Copied sample {sample.name} and its label.")
            else:
                print(f"  - Failed to copy sample {sample.name}.")
