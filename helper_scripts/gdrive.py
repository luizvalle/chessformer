import pandas as pd
import io

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

class GDrive():
    def __init__(self, service_account_json_key_path):
        scope = ["https://www.googleapis.com/auth/drive"]
        credentials = service_account.Credentials.from_service_account_file(
            filename=service_account_json_key_path,
            scopes=scope)
        self.service = build("drive", "v3", credentials=credentials)

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
            buf, mimetype="application/octet-stream", chunksize=2048*1024, resumable=True)
        request = self.service.files().create(media_body=media, body=metadata)
        response = None
        while response is None:
            status, response = request.next_chunk()

    def is_file(self, parent_directory_id, file_name):
        response = self.service.files().list(
            q=f"name = '{file_name}' and mimeType = 'application/octet-stream' and '{parent_directory_id}' in parents and trashed = False"
        ).execute()
        return len(response.get("files", [])) > 0
    
    def get_files(self, parent_directory_id):
        response = self.service.files().list(
            q=f"mimeType = 'application/octet-stream' and '{parent_directory_id}' in parents and trashed = False",
            spaces="drive",
            fields="files(name)"
        ).execute()
        file_names = {returned_file.get("name") for returned_file in response.get("files", [])}
        return file_names

