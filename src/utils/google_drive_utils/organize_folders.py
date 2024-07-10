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
    source_folder_id = '1sH7jg7hjWo_Z55FQ85m9NmKEHvd9LROq'  # Source folder ID in Google Drive
    dest_folder_id = '1aRXDDVstxwS8boo9xdlXj6XywSeYDEnQ'  # Destination folder ID in Google Drive

    flower_ids = [256, 257, 246, 266, 241, 251, 261, 262, 242, 247, 
                260, 268, 267, 263, 245, 270, 252, 255, 265, 243,
                253, 269, 264, 249, 250, 258, 254, 248, 244, 259, ]
    image_ids = [8866, 8872, 8879, 8885, 8892, 8898, 8906, 8915, 8922, 8933, 
                8940, 8949, 8956, 8963, 8972, 8980, 8987, 8995, 9004, 9012, 
                9021, 9031, 9038, 9050, 9058, 9066, 9075, 9085, 9092, 9100,
                ]

    flower_id_images = [f'IMG_{image_id}.JPG' for image_id in image_ids]

    organize_images(service, source_folder_id, dest_folder_id, flower_ids, flower_id_images)
    
    # # validate the number of images in each list
    # print(len(flower_ids))
    # # print(len(image_ids))
    
    # # validate that there is no duplicates in flower_ids
    # print(len(set(flower_ids)))
    
    # # check which number is missing from 241 to 270
    # print(set(range(241, 271)) - set(flower_ids))
    
