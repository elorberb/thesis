from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import os

# Path to the service account key file
SERVICE_ACCOUNT_FILE = '/home/etaylor/code_projects/thesis/src/app/trichome-classification-study.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Authenticate using the service account
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)

# Create the service
service = build('drive', 'v3', credentials=credentials)


def list_files_in_folder(service, folder_id):
    """List all files in a Google Drive folder."""
    items = []
    query = f"'{folder_id}' in parents and trashed=false"
    page_token = None

    while True:
        response = service.files().list(q=query,
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name, mimeType, parents)',
                                        pageToken=page_token).execute()
        items.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return items

def create_folder(service, parent_folder_id, folder_name):
    """Create a folder in Google Drive."""
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_folder_id]
    }
    folder = service.files().create(body=file_metadata, fields='id').execute()
    return folder.get('id')


def move_file(service, file_id, new_folder_id):
    """Move a file to a new folder in Google Drive."""
    file = service.files().get(fileId=file_id, fields='parents').execute()
    previous_parents = ",".join(file.get('parents'))
    file = service.files().update(fileId=file_id,
                                addParents=new_folder_id,
                                removeParents=previous_parents,
                                fields='id, parents').execute()


def copy_file(service, file_id, new_folder_id, new_name=None):
    """Copy a file to a new folder in Google Drive."""
    file_metadata = {'parents': [new_folder_id]}
    if new_name:
        file_metadata['name'] = new_name
    copied_file = service.files().copy(fileId=file_id, body=file_metadata).execute()
    return copied_file.get('id')


def organize_images(service, source_folder_id, dest_folder_id, flower_ids, flower_id_images):
    items = list_files_in_folder(service, source_folder_id)
    
    current_flower_index = 0
    current_flower_id = flower_ids[current_flower_index]
    flower_folders = {}

    for item in sorted(items, key=lambda x: x['name']):
        file_id = item['id']
        file_name = item['name']

        if file_name == flower_id_images[current_flower_index]:
            current_flower_id = flower_ids[current_flower_index]
            if current_flower_id not in flower_folders:
                flower_folder_id = create_folder(service, dest_folder_id, str(current_flower_id))
                flower_folders[current_flower_id] = flower_folder_id
                
            # Copy the image representing the flower ID to the flower's folder
            flower_folder_id = flower_folders[current_flower_id]
            copy_file(service, file_id, flower_folder_id)
            print(f"Copied flower ID image {file_name} to folder {current_flower_id}")
            current_flower_index += 1
            if current_flower_index >= len(flower_ids):
                break
        else:
            flower_folder_id = flower_folders[current_flower_id]
            copy_file(service, file_id, flower_folder_id)
            print(f"Copied {file_name} to folder {current_flower_id}")


if __name__ == '__main__':
    source_folder_id = '104rOQXe3se35zDNmoz-ARVBwT3YboADX'  # Source folder ID in Google Drive
    dest_folder_id = '1FJTRl_hnKDxAnoX2tOsH-vb2wvgk84Ri'  # Destination folder ID in Google Drive

    flower_ids = [203, 200, 191, 205, 188, 195, 186, 187, 190, 201, 
                192, 196, 181, 206, 208, 197, 207, 184, 185, 210,
                199, 193, 194, 202, 189, 183, 204, 209, 198, ]
    image_ids = [8140, 8150, 8160, 8170, 8178, 8189, 8199, 8205, 8217, 8232, 
                8252, 8260, 8271, 8282, 8290, 8302, 8312, 8322, 8333, 8345,
                8356, 8369, 8380, 8389, 8398, 8407, 8417, 8427, 8435, ]

    flower_id_images = [f'IMG_{image_id}.JPG' for image_id in image_ids]

    organize_images(service, source_folder_id, dest_folder_id, flower_ids, flower_id_images)
    
    # validate the number of images in each list
    # print(len(flower_ids))
    # print(len(image_ids))
    
    # # validate that there is no duplicates in flower_ids
    # print(len(set(flower_ids)))
    
