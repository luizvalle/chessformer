import chess.pgn
import zstandard as zstd
import requests
import io
import pandas as pd
import re
import multiprocessing

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


DOWNLOAD_LIST = "https://database.lichess.org/standard/list.txt"
PARENT_DIR_ID = "1Cwnlq0ziqLP6h0LsZlrzs3lLTHGRDPLI"
CREDENTIALS_JSON = "./credentials.json"  # File with gDrive credentials
CHUNK_SIZE = 5
N_PROCESSES = multiprocessing.cpu_count() + 2


class CompressedPgnHeaderIterator:
    def __init__(self, download_link):
        dctx = zstd.ZstdDecompressor()
        # Stream the results so we do not load everything
        # into memory at once
        response = requests.get(url=download_link, stream=True)
        reader = dctx.stream_reader(response.raw)
        self.text_stream = io.TextIOWrapper(reader, encoding='utf-8')

    def __iter__(self):
        return self

    def __next__(self):
        header = chess.pgn.read_headers(self.text_stream)
        if header:
            return header
        else:
            raise StopIteration


class GDrive():
    def __init__(self, service_account_json_key_path):
        scope = ["https://www.googleapis.com/auth/drive"]
        credentials = service_account.Credentials.from_service_account_file(
            filename=service_account_json_key_path,
            scopes=scope)
        self.service = build('drive', 'v3', credentials=credentials)

    def write_dataframe(self, df, parent_directory_id, file_name, compression="zstd"):
        buf = io.BytesIO()
        original_close = buf.close
        buf.close = lambda: None  # Hack to keep to_parquet from closing BytesIO
        try:
            df.to_parquet(buf, compression=compression, closed=False)
        finally:
            buf.close = original_close
        metadata = {"name": f"{file_name}.{compression}",
                    "parents": [parent_directory_id]}
        media = MediaIoBaseUpload(
            buf, mimetype="application/octet-stream", chunksize=5e6, resumable=True)
        request = self.service.files().create(media_body=media, body=metadata)
        response = None
        while response is None:
            status, response = request.next_chunk()

    def is_file(self, parent_directory_id, file_name):
        response = self.service.files().list(
            q=f"name = '{file_name}' and mimeType = 'application/octet-stream' and '{parent_directory_id}' in parents and trashed = False"
        ).execute()
        return len(response.get('files', [])) > 0


def file_name_from_link(download_link):
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1)
    path = f"{date}.parquet.zstd"
    return path


def process_headers(gDrive, download_link):
    column_types = {
        "Event": "category",
        "Result": "category",
        "WhiteElo": "uint16",
        "BlackElo": "uint16",
        "TimeControl": "category",
        "Termination": "category"
    }
    print(download_link)
    new_file_name = file_name_from_link(download_link)
    headers = CompressedPgnHeaderIterator(download_link)
    games_info = [{
        "Event": header["Event"],
        "Result": header["Result"],
        "WhiteElo": int(header["WhiteElo"]) if header["WhiteElo"].isnumeric() else 0,
        "BlackElo": int(header["BlackElo"]) if header["BlackElo"].isnumeric() else 0,
        "TimeControl": header["TimeControl"],
        "Termination": header["Termination"]}
        for header in headers]
    df = pd.DataFrame(data=games_info).astype(column_types)
    gDrive.write_dataframe(df, PARENT_DIR_ID, new_file_name)


def main():
    gDrive = GDrive(CREDENTIALS_JSON)

    download_links = sorted(
        requests.get(DOWNLOAD_LIST).text.split('\n'),
        reverse=False)

    print(f"Found {len(download_links)} files to download.")
    print("Checking how many were already processed...")

    unprocessed_links = list()
    for download_link in tqdm(download_links):
        file_name = file_name_from_link(download_link)
        if not gDrive.is_file(PARENT_DIR_ID, file_name):
            unprocessed_links.append(download_link)

    print(f"{len(download_links) - len(unprocessed_links)} files already processed.")
    print(f"Processing remaining {len(unprocessed_links)} files...")

    bound_process_headers = partial(process_headers, gDrive)

    with Pool(processes=N_PROCESSES) as pool:
        result = pool.imap_unordered(
            func=bound_process_headers,
            iterable=unprocessed_links,
            chunksize=CHUNK_SIZE)
        for _ in tqdm(result, total=len(unprocessed_links)):
            pass


if __name__ == "__main__":
    main()
