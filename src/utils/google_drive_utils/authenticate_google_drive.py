from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
import os
import io

# Path to the OAuth2 credentials file
CLIENT_SECRET_FILE = "/home/etaylor/code_projects/thesis/src/utils/google_drive_utils/creds/client_secret_261183461521-ag57t3cdlmeadm0oqo6tq3qhv477f67o.apps.googleusercontent.com.json"

# Define the scopes
SCOPES = ["https://www.googleapis.com/auth/drive"]

# Authenticate using OAuth2
flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
creds = flow.run_console()  # Use console-based authentication flow

# Create the service
service = build("drive", "v3", credentials=creds)

# Save credentials
TOKEN_PATH = (
    "/home/etaylor/code_projects/thesis/src/utils/google_drive_utils/creds/token.json"
)
with open(TOKEN_PATH, "w") as token_file:
    token_file.write(creds.to_json())
