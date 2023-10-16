import pandas as pd
import re
import requests
import multiprocess

from chess_iterators import CompressedPgnHeaderIterator
from gdrive import GDrive
from tqdm.auto import tqdm
from multiprocess import Pool


DOWNLOAD_LIST = "https://database.lichess.org/standard/list.txt"
PARENT_DIR_ID = "1Cwnlq0ziqLP6h0LsZlrzs3lLTHGRDPLI"
CREDENTIALS_JSON = "./credentials.json"  # File with gDrive credentials
N_PROCESSES = 5


def file_name_from_link(download_link):
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1)
    path = f"{date}.parquet"
    return path


def process_headers(arg):
    i, download_link = arg
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
    games_info = list()
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1).strip()
    with tqdm(total=headers.total_num_bytes(), position=i + 1, unit='B', unit_scale=True, leave=False, desc=date, ncols=80) as pbar:
        try:
            for header in headers:
                info = {
                        "Event": header["Event"],
                        "Result": header["Result"],
                        "WhiteElo": int(header["WhiteElo"]) if header["WhiteElo"].isnumeric() else 0,
                        "BlackElo": int(header["BlackElo"]) if header["BlackElo"].isnumeric() else 0,
                        "TimeControl": header["TimeControl"],
                        "Termination": header["Termination"]
                        }
                games_info.append(info)
                update_value = headers.total_num_bytes_read() - pbar.n
                update_value = update_value if pbar.n + update_value <= headers.total_num_bytes() else headers.total_num_bytes() - pbar.n
                pbar.update(update_value)
            df = pd.DataFrame(data=games_info).astype(column_types)
            gDrive = GDrive(CREDENTIALS_JSON)
            gDrive.write_dataframe(df, PARENT_DIR_ID, new_file_name)
        except Exception as e:
            print(e)


def main():
    gDrive = GDrive(CREDENTIALS_JSON)

    download_links = sorted(
        requests.get(DOWNLOAD_LIST).text.split('\n'),
        reverse=False)

    print(f"Found {len(download_links)} files to download.", flush=True)
    print("Checking how many were already processed...", flush=True)

    existing_files = gDrive.get_files(PARENT_DIR_ID)
    unprocessed_links = [download_link for download_link in download_links
            if f"{file_name_from_link(download_link)}.zstd" not in existing_files]
    unprocessed_links = list(enumerate(unprocessed_links)) # Add index so that tqdm can place the progress bar

    print(f"{len(download_links) - len(unprocessed_links)} files already processed.", flush=True)
    print(f"Processing remaining {len(unprocessed_links)} files...", flush=True)
    print(f"Using {N_PROCESSES} processes.")

    chunk_size = 1
    with Pool(N_PROCESSES) as pool:
        list(tqdm(pool.imap_unordered(process_headers, unprocessed_links, chunksize=chunk_size), desc="Overall progress", total=len(unprocessed_links), position=0, ncols=80, leave=True))


if __name__ == "__main__":
    main()
