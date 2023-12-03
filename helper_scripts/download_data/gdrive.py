import pandas as pd
import io
import os

from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload


class GDrive():
    def __init__(self, json_key_path, is_service_account=False):
        scopes = ["https://www.googleapis.com/auth/drive"]
        if is_service_account:
            credentials = GDrive._get_service_account_credentials(
                    json_key_path, scopes)
        else:
            credentials = GDrive._get_user_account_credentials(
                    json_key_path, scopes)
        self.service = build("drive", "v3", credentials=credentials)

    @staticmethod
    def _get_service_account_credentials(json_key_path, scopes):
        credentials = service_account.Credentials.from_service_account_file(
            filename=json_key_path,
            scopes=scopes)
        return credentials

    @staticmethod
    def _get_user_account_credentials(json_key_path, scopes):
        credentials = None
        token_file_name = "./token.json"
        if os.path.exists(token_file_name):
            credentials = Credentials.from_authorized_user_file(
                    token_file_name, scopes)
        if not credentials or not credentials.valid:
            if (credentials and credentials.expired
                and credentials.refresh_token):
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                        json_key_path, scopes)
                credentials = flow.run_local_server(port=0)
            # Save credentials for the next run
            with open(token_file_name, "w") as token_file:
                token_file.write(credentials.to_json())
        return credentials

    def write_dataframe(self, df, parent_directory_id, file_name, compression="zstd"):
        buf = io.BytesIO()
        original_close = buf.close
        buf.close = lambda: None  # Hack to keep to_parquet from closing BytesIO
        try:
            df.to_parquet(buf, compression=compression)
        finally:
            buf.close = original_close
        metadata = {
                "name": f"{file_name}.{compression}",
                "parents": [parent_directory_id]
                }
        # Chunk size has to be a multiple of 256 but be below 5MB
        media = MediaIoBaseUpload(
            buf,
            mimetype="application/octet-stream",
            chunksize=2048*1024,
            resumable=True)
        self._write_resumable(metadata, media)

    def write_file(self, source_file_name, parent_directory_id, out_file_name):
        metadata = {
                "name": out_file_name,
                "parents": [parent_directory_id]
                }
        # Chunk size has to be a multiple of 256 and be below 5MB
        media = MediaFileUpload(
            source_file_name,
            mimetype="application/octet-stream",
            chunksize=2048*1024,
            resumable=True)
        file_id = self._write_resumable(metadata, media)

    def _write_resumable(self, metadata, media):
        request = self.service.files().create(
                media_body=media, body=metadata, fields="id")
        response = None
        while response is None:
            status, response = request.next_chunk()
        return response.get("id")

    def _set_permission(self, file_id, type, role, email_address):
        permission = {
                "type": type,
                "role": role,
                "emailAddress": email_address
                }
        self.service.permissions().create(
                fileId=file_id, body=permission,
                transferOwnership=True).execute()

    def is_file(self, parent_directory_id, file_name):
        response = self.service.files().list(
            q=f"name = '{file_name}' and mimeType = 'application/octet-stream' and '{parent_directory_id}' in parents and trashed = False"
        ).execute()
        return len(response.get("files", [])) > 0
    
    def get_files(self, parent_directory_id):
        file_names = set()
        page_token = ""
        while page_token is not None:
            response = self.service.files().list(
                    q=f"mimeType = 'application/octet-stream' and '{parent_directory_id}' in parents and trashed = False",
                    pageSize=1000,
                    pageToken=page_token,
                    spaces="drive",
                    fields="nextPageToken, files(name)"
                    ).execute()
            new_file_names = {file.get("name") for file in response.get("files", [])}
            file_names.update(new_file_names)
            page_token = response.get("nextPageToken")
        return file_names

