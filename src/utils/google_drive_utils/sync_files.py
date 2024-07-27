from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import os
import io

# Path to the OAuth2 credentials file
CLIENT_SECRET_FILE = '/home/etaylor/code_projects/thesis/src/utils/google_drive_utils/creds/client_secret_261183461521-ag57t3cdlmeadm0oqo6tq3qhv477f67o.apps.googleusercontent.com.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive']

# Authenticate using OAuth2
flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
creds = flow.run_console()  # Use console-based authentication flow

# Create the service
service = build('drive', 'v3', credentials=creds)

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

def delete_local_folder(path):
    """Delete a folder and its contents."""
    if not os.path.isdir(path):
        print(f"Provided path: {path} is not a directory.")
        return
    
    for root, dirs, files in os.walk(path, topdown=False):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
                print(f"Deleted folder: {dir_path}")
            except Exception as e:
                print(f"Error deleting folder {dir_path}: {e}")
    try:
        os.rmdir(path)
        print(f"Deleted top-level folder: {path}")
    except Exception as e:
        print(f"Error deleting top-level folder {path}: {e}")

def upload_file(service, file_path, folder_id):
    """Upload a file to Google Drive."""
    file_name = os.path.basename(file_path)
    query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    if response.get('files', []):
        print(f"File already exists: {file_name}. Skipping upload.")
        return

    media = MediaFileUpload(file_path, resumable=True)
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    print(f"Uploaded file: {file_name} with ID: {file.get('id')}")

def create_folder(service, folder_name, parent_folder_id):
    """Create a folder in Google Drive."""
    query = f"name='{folder_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
    response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    if response.get('files', []):
        print(f"Folder already exists: {folder_name}. Using existing folder.")
        return response['files'][0]['id']

    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder',
        'parents': [parent_folder_id]
    }
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    print(f"Created folder: {folder_name} with ID: {folder.get('id')}")
    return folder.get('id')

def upload_folder_dfs(service, local_path, parent_folder_id):
    """Upload all contents of a folder to Google Drive using DFS, preserving the folder structure."""
    stack = [(local_path, parent_folder_id)]

    while stack:
        current_local_path, current_parent_id = stack.pop()
        current_folder_id = current_parent_id

        for entry in os.listdir(current_local_path):
            full_path = os.path.join(current_local_path, entry)
            if os.path.isdir(full_path):
                new_folder_id = create_folder(service, entry, current_folder_id)
                stack.append((full_path, new_folder_id))
            else:
                upload_file(service, full_path, current_folder_id)

def check_storage_quota(service):
    """Check the storage quota for the service account."""
    about = service.about().get(fields="storageQuota").execute()
    quota = about.get('storageQuota', {})
    usage = int(quota.get('usage', 0))
    limit = int(quota.get('limit', 0))
    print(f"Storage used: {usage / (1024**3):.2f} GB")
    print(f"Storage limit: {limit / (1024**3):.2f} GB")
    if limit > 0:
        print(f"Storage usage percentage: {usage / limit * 100:.2f}%")
    else:
        print("Unlimited storage")

if __name__ == '__main__':
    # folder_id = '1-cKSnT-qIwW3_nq6KkyhaZHrVCvEcvss'  # folder id in google drive
    # local_download_path = '/sise/shanigu-group/etaylor/assessing_cannabis_exp/images/day_7_2024_06_20/lab/182'  # images path in the cluster

    # # Ensure the local download directory exists
    # if not os.path.exists(local_download_path):
    #     os.makedirs(local_download_path)

    # overwrite_existing_files = False  # Set to True to overwrite existing files, False to skip them
    # download_folder(service, folder_id, local_download_path, overwrite_existing_files)
    
    # # in order to clean empty folders
    # folder_to_clean = local_download_path
    # delete_empty_folders(folder_to_clean)
    
    # ---------- Upload folder - for uploading the results to the google drive
    local_upload_path = '/sise/shanigu-group/etaylor/assessing_cannabis_exp/results/day_1_2024_05_30'  # Specify your local folder path here
    upload_folder_id = '1TVEFWNU1Mt9T5epE9_7Ta-kLTBByty-9'  # Specify the Google Drive folder ID here
    upload_folder_dfs(service, local_upload_path, upload_folder_id)
    print("Upload completed.")
    # ---------- check storage quata -----------------
    # check_storage_quota(service)
