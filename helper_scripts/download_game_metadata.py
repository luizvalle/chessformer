import pandas as pd
import re
import requests
import multiprocess

from chess_iterators import CompressedPgnHeaderIterator
from gdrive import GDrive
from tqdm import tqdm
from multiprocess import Pool


DOWNLOAD_LIST = "https://database.lichess.org/standard/list.txt"
PARENT_DIR_ID = "1Cwnlq0ziqLP6h0LsZlrzs3lLTHGRDPLI"
CREDENTIALS_JSON = "./credentials.json"  # File with gDrive credentials
N_PROCESSES = multiprocess.cpu_count() - 1 if multiprocess.cpu_count() > 1 else 1


def file_name_from_link(download_link):
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1)
    path = f"{date}.parquet"
    return path


def process_headers(download_link):
    column_types = {
            "Event": "category",
            "Result": "category",
            "WhiteElo": "uint16",
            "BlackElo": "uint16",
            "TimeControl": "category",
            "Termination": "category"
            }
    new_file_name = file_name_from_link(download_link)
    headers = CompressedPgnHeaderIterator(download_link)
    try:
        games_info = [{
            "Event": header["Event"],
            "Result": header["Result"],
            "WhiteElo": int(header["WhiteElo"]) if header["WhiteElo"].isnumeric() else 0,
            "BlackElo": int(header["BlackElo"]) if header["BlackElo"].isnumeric() else 0,
            "TimeControl": header["TimeControl"],
            "Termination": header["Termination"]}
            for header in headers]
        df = pd.DataFrame(data=games_info).astype(column_types)
        gDrive = GDrive(CREDENTIALS_JSON)
        gDrive.write_dataframe(df, PARENT_DIR_ID, new_file_name)
    except Exception as e:
        print(f"Error processing {new_file_name}: {e}")


def main():
    gDrive = GDrive(CREDENTIALS_JSON)

    download_links = sorted(
        requests.get(DOWNLOAD_LIST).text.split('\n'),
        reverse=False)

    print(f"Found {len(download_links)} files to download.")
    print("Checking how many were already processed...")

    existing_files = gDrive.get_files(PARENT_DIR_ID)
    unprocessed_links = [download_link for download_link in download_links
            if f"{file_name_from_link(download_link)}.zstd" not in existing_files]

    print(f"{len(download_links) - len(unprocessed_links)} files already processed.")
    print(f"Processing remaining {len(unprocessed_links)} files...")
    print(f"Using {N_PROCESSES} processes.")
    chunk_size = 1
    with Pool(N_PROCESSES) as pool:
        list(tqdm(pool.imap(process_headers, unprocessed_links, chunksize=chunk_size), total=len(unprocessed_links)))


if __name__ == "__main__":
    main()

