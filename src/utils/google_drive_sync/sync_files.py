from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import os
import io

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
    """List all files and folders in a Google Drive folder."""
    items = []
    query = f"'{folder_id}' in parents and trashed=false"
    page_token = None

    while True:
        response = service.files().list(q=query,
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name, mimeType)',
                                        pageToken=page_token).execute()
        items.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break
    return items


def download_file(service, file_id, file_path, overwrite=False):
    """Download a file from Google Drive."""
    if not overwrite and os.path.exists(file_path):
        print(f"Skipping existing file: {file_path}")
        return
    print(f"Downloading file: {file_path}")
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Download {int(status.progress() * 100)}% for file {file_path}.")
    print(f"Downloaded file to {file_path}")


def download_folder(service, folder_id, local_path, overwrite=False):
    """Download all contents of a folder recursively."""
    print(f"Accessing folder: {local_path}")
    items = list_files_in_folder(service, folder_id)
    for item in items:
        item_name = item['name']
        item_id = item['id']
        item_mime_type = item['mimeType']
        local_item_path = os.path.join(local_path, item_name)

        if item_mime_type == 'application/vnd.google-apps.folder':
            # Create local folder and download its contents
            if not os.path.exists(local_item_path):
                os.makedirs(local_item_path)
            download_folder(service, item_id, local_item_path, overwrite)
        elif item_mime_type.startswith('application/vnd.google-apps.'):
            # Export Google Docs editors files (skip in this example)
            print(f"Skipping Google Docs editors file: {item_name}")
        else:
            download_file(service, item_id, local_item_path, overwrite)
            
            
def delete_empty_folders(path):
    """Recursively delete empty folders."""
    if not os.path.isdir(path):
        print(f"Provided path: {path} is not a directory.")
        return
    
    # Check all subfolders
    for root, dirs, files in os.walk(path, topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                if not os.listdir(dir_path):  # Check if the directory is empty
                    os.rmdir(dir_path)
                    print(f"Deleted empty folder: {dir_path}")
            except Exception as e:
                print(f"Error deleting folder {dir_path}: {e}")

    # Check the top-level folder
    try:
        if not os.listdir(path):  # Check if the top-level directory is empty
            os.rmdir(path)
            print(f"Deleted top-level empty folder: {path}")
    except Exception as e:
        print(f"Error deleting top-level folder {path}: {e}")

if __name__ == '__main__':
    # folder_id = '1nU_mahoiGMzifPFItmM8iv9OlaCsWP2n'  # assessing_cannabis_experiment folder in google drive
    # local_download_path = '/home/etaylor/images/assessing_cannabis_experiment_images'  # images path in the cluster

    # # Ensure the local download directory exists
    # if not os.path.exists(local_download_path):
    #     os.makedirs(local_download_path)

    # overwrite_existing_files = False  # Set to True to overwrite existing files, False to skip them
    # download_folder(service, folder_id, local_download_path, overwrite_existing_files)
    
    # in order to clean empty folders
    folder_to_clean = "/home/etaylor/images/assessing_cannabis_experiment_images/day_1_2024_05_30"
    delete_empty_folders(folder_to_clean)
