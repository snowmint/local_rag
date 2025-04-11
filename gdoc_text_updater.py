# gdoc_text_updater.py (OAuth client secret version)

import os
import re
import json
import pickle
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# ====== CONFIGURATION ======
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
CLIENT_SECRET_FILE = './credentials/client_secret.json'  # OAuth client secret file
# Stores user access/refresh token avoid login every time execute the code
TOKEN_FILE = './cache_log/token.pickle'
# Replace with your Google Drive folder ID
FOLDER_ID = '1-rJI9CmBtIyjuyVyq5NeWvlwxnzcQytl'
# Store last modified timestamps
LAST_MODIFIED_TRACKER = './cache_log/file_modified_cache.json'

# ====== GOOGLE DOCS SECTION ======

data_save_path = './data/'


def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)


def fetch_gdoc_list():
    service = get_drive_service()
    try:
        results = service.files().list(
            q=f"'{FOLDER_ID}' in parents and mimeType='application/vnd.google-apps.document'",
            pageSize=100,
            fields="files(id, name, modifiedTime)"
        ).execute()
        return results.get('files', [])
    except HttpError as error:
        print(f"An error occurred: {error}")
        return []


def export_gdoc_as_text(file_id):
    service = get_drive_service()
    try:
        response = service.files().export(fileId=file_id, mimeType='text/plain').execute()
        return response.decode('utf-8')
    except HttpError as error:
        print(f"Error exporting Google Doc: {error}")
        return None

# ====== UTILITY ======


def clean_filename(name):
    return re.sub(r'[\\/*?\"<>|]', '_', name)


def load_mod_time_cache():
    if os.path.exists(LAST_MODIFIED_TRACKER):
        with open(LAST_MODIFIED_TRACKER, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_mod_time_cache(cache):
    with open(LAST_MODIFIED_TRACKER, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)


if __name__ == '__main__':
    mod_time_cache = load_mod_time_cache()
    updated_cache = {}

    docs = fetch_gdoc_list()
    print(f"Found {len(docs)} Google Docs in the folder.")
    for doc in docs:
        doc_id = doc['id']
        name = doc['name']
        mod_time = doc['modifiedTime']

        updated_cache[doc_id] = mod_time

        if doc_id in mod_time_cache and mod_time_cache[doc_id] == mod_time:
            print(f"Skipping unchanged: {name}")
            continue

        print(f"Processing updated/new file: {name}")
        content = export_gdoc_as_text(doc_id)
        filename = clean_filename(name) + '.txt'
        if content:
            with open(data_save_path + filename, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Saved: {filename}")

    save_mod_time_cache(updated_cache)
