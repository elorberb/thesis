from googleapiclient.discovery import build
import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials

# Path to the OAuth2 credentials file
CLIENT_SECRET_FILE = "/home/etaylor/code_projects/thesis/src/utils/google_drive_utils/creds/client_secret_261183461521-ag57t3cdlmeadm0oqo6tq3qhv477f67o.apps.googleusercontent.com.json"

# Define the scopes
SCOPES = ["https://www.googleapis.com/auth/drive"]

TOKEN_PATH = (
    "/home/etaylor/code_projects/thesis/src/utils/google_drive_utils/creds/token.json"
)

# Define the scopes
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Load credentials
if os.path.exists(TOKEN_PATH):
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
else:
    # Authenticate using OAuth2
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
    creds = flow.run_console()  # Use console-based authentication flow

# Create the service
service = build("drive", "v3", credentials=creds)


def list_files_in_folder(service, folder_id):
    """List all files in a Google Drive folder."""
    items = []
    query = f"'{folder_id}' in parents and trashed=false"
    page_token = None

    while True:
        response = (
            service.files()
            .list(
                q=query,
                spaces="drive",
                fields="nextPageToken, files(id, name, mimeType, parents)",
                pageToken=page_token,
            )
            .execute()
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken", None)
        if page_token is None:
            break
    return items


def create_folder(service, parent_folder_id, folder_name):
    """Create a folder in Google Drive."""
    file_metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    folder = service.files().create(body=file_metadata, fields="id").execute()
    return folder.get("id")


def move_file(service, file_id, new_folder_id):
    """Move a file to a new folder in Google Drive."""
    file = service.files().get(fileId=file_id, fields="parents").execute()
    previous_parents = ",".join(file.get("parents"))
    file = (
        service.files()
        .update(
            fileId=file_id,
            addParents=new_folder_id,
            removeParents=previous_parents,
            fields="id, parents",
        )
        .execute()
    )


def copy_file(service, file_id, new_folder_id, new_name=None):
    """Copy a file to a new folder in Google Drive."""
    file_metadata = {"parents": [new_folder_id]}
    if new_name:
        file_metadata["name"] = new_name
    copied_file = service.files().copy(fileId=file_id, body=file_metadata).execute()
    return copied_file.get("id")


def extract_image_number(file_name):
    try:
        return int(file_name.split("_")[1].split(".")[0])
    except (IndexError, ValueError):
        return None


def organize_images_deprecated(
    service, source_folder_id, dest_folder_id, flower_ids, flower_id_images
):
    items = list_files_in_folder(service, source_folder_id)

    # Extract numeric parts of image names and determine the range
    image_numbers = [int(name.split("_")[1].split(".")[0]) for name in flower_id_images]
    min_image_num = min(image_numbers)
    max_image_num = max(image_numbers)

    current_flower_index = 0
    current_flower_id = flower_ids[current_flower_index]
    flower_folders = {}

    for item in sorted(items, key=lambda x: x["name"]):
        file_id = item["id"]
        file_name = item["name"]

        image_number = extract_image_number(file_name)
        if image_number is not None and min_image_num <= image_number <= max_image_num:
            if file_name == flower_id_images[current_flower_index]:
                current_flower_id = flower_ids[current_flower_index]
                if current_flower_id not in flower_folders:
                    flower_folder_id = create_folder(
                        service, dest_folder_id, str(current_flower_id)
                    )
                    flower_folders[current_flower_id] = flower_folder_id

                flower_folder_id = flower_folders[current_flower_id]
                copy_file(service, file_id, flower_folder_id)
                print(
                    f"Copied flower ID image {file_name} to folder {current_flower_id}"
                )
                current_flower_index += 1
                if current_flower_index >= len(flower_ids):
                    break
            else:
                flower_folder_id = flower_folders[current_flower_id]
                copy_file(service, file_id, flower_folder_id)
                print(f"Copied {file_name} to folder {current_flower_id}")


def organize_images(
    service, source_folder_id, dest_folder_id, flower_ids, flower_id_images
):
    items_sorted = sorted(
        list_files_in_folder(service, source_folder_id), key=lambda x: x["name"]
    )
    flower_folders = {}

    image_numbers = [extract_image_number(name) for name in flower_id_images]

    for index, flower_id in enumerate(flower_ids):
        # Determine the image range for the current flower
        start_num = image_numbers[index]
        end_num = (
            image_numbers[index + 1] - 1
            if index + 1 < len(image_numbers)
            else extract_image_number(items_sorted[-1]["name"])
        )

        # Create folder if it doesn't exist
        flower_folder_id = flower_folders.get(flower_id)
        if not flower_folder_id:
            flower_folder_id = create_folder(service, dest_folder_id, str(flower_id))
            flower_folders[flower_id] = flower_folder_id

        # Copy images within the determined range to the flower folder
        copy_images_in_range(
            service, items_sorted, start_num, end_num, flower_folder_id, str(flower_id)
        )


def copy_images_in_range(service, items, start_num, end_num, folder_id, flower_id: str):
    for item in items:
        image_number = extract_image_number(item["name"])
        if image_number is not None and start_num <= image_number <= end_num:
            copy_file(service, item["id"], folder_id)
            print(f"Copied {item['name']} to folder {folder_id} ({flower_id})")


def create_main_and_subfolders(service, parent_folder_id, main_folder_name):
    """
    Create a main folder and 4 subfolders ('lab_unorganized', 'lab', 'greenhouse_unorganized', 'greenhouse') in Google Drive.

    Args:
        service: Google Drive API service instance.
        parent_folder_id: ID of the parent folder where the main folder will be created.
        main_folder_name: Name of the main folder to create.

    Returns:
        Dictionary containing IDs of the main folder and its subfolders.
    """
    # Create the main folder
    file_metadata = {
        "name": main_folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_folder_id],
    }
    main_folder = service.files().create(body=file_metadata, fields="id").execute()
    main_folder_id = main_folder["id"]

    print(f"Created main folder: {main_folder_name} (ID: {main_folder_id})")

    # Subfolder names
    subfolder_names = ["lab_unorganized", "lab", "greenhouse_unorganized", "greenhouse"]

    # Create subfolders inside the main folder
    subfolder_ids = {}
    for subfolder_name in subfolder_names:
        subfolder_metadata = {
            "name": subfolder_name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [main_folder_id],
        }
        subfolder = (
            service.files().create(body=subfolder_metadata, fields="id").execute()
        )
        subfolder_ids[subfolder_name] = subfolder["id"]
        print(
            f"Created subfolder: {subfolder_name} (ID: {subfolder_ids[subfolder_name]})"
        )

    return {
        "main_folder_id": main_folder_id,
        "subfolders": subfolder_ids,
    }


def check_how_many_jpg_files_in_folder(folder_path):
    # check how many JPG files there are
    import os

    count = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".JPG"):
                count += 1

    print(count)


if __name__ == "__main__":

    # ------------------- Code for Organize the images to folders -------------------
    source_folder_id = (
        "1jXGDvVvqlLQQSHUFnl3WSAB02y-q2MBz"  # Source folder ID in Google Drive
    )
    dest_folder_id = (
        "1Kfgw-L-R-aDS2rNi20Kfh8UqY1Vtk1_1"  # Destination folder ID in Google Drive
    )

    flower_ids = list(range(136, 151))
    image_ids = [
        4576,
        4604,
        4630,
        4658,
        4684,
        4719,
        4747,
        4783,
        4813,
        4837,
        4867,
        4894,
        4920,
        4951,
        4977,
    ]

    flower_id_images = [f"IMG_{image_id}.JPG" for image_id in image_ids]

    print(flower_id_images)

    organize_images(
        service, source_folder_id, dest_folder_id, flower_ids, flower_id_images
    )

    # ------------------- Code for Create main and subfolders -------------------
    # parent_folder_id = (
    #     "1sCNQ_PBeFqA9fHgXEPG41CUZczUKnG3X"  # experiment_2 images folder ID in Google Drive
    # )
    # main_folder_name = "day_4_2024_12_17"  # Name of the main folder to create
    # create_main_and_subfolders(service, parent_folder_id, main_folder_name)

    # ---------------------- Code for check how many JPG files in a folder -------------------
    # folder_path = "/sise/shanigu-group/etaylor/assessing_cannabis_exp/experiment_2/images/day_1_2024_12_05/lab"
    # check_how_many_jpg_files_in_folder(folder_path)
