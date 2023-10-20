# Usage: python3 download_game_metadata.py
#
# Iterates through the headers of the PGN files and constructs a dataframe from
# them. This dataframe is then stored in Google Drive.
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import re
import requests
import multiprocess
import sys
import os
import shutil

from chess_iterators import CompressedPgnHeaderIterator
from gdrive import GDrive
from tqdm.auto import tqdm
from multiprocess import Pool, RLock, freeze_support, current_process
from datetime import datetime
from queue import Queue
from concurrent.futures import ThreadPoolExecutor


DOWNLOAD_LIST = "https://database.lichess.org/standard/list.txt"
PARENT_DIR_ID = "1Cwnlq0ziqLP6h0LsZlrzs3lLTHGRDPLI"
CREDENTIALS_JSON = "./credentials.json"  # File with gDrive credentials
SCRATCH_DIR = "./.scratch" # Where temporary data will be stored
N_PROCESSES = 4
MAX_QUEUE_SIZE = 1e6
MAX_BUFFER_LEN = 1e6


def file_name_from_link(download_link):
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1)
    path = f"{date}.parquet"
    return path


def record_producer(download_link, queue):
    headers = CompressedPgnHeaderIterator(download_link)
    tqdm_pos = current_process()._identity[0] # Index of the current process
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1).strip()
    pbar = tqdm(total=headers.total_num_bytes(),
                unit='B',
                position=tqdm_pos,
                unit_scale=True,
                leave=False,
                desc=f"{date} (download)",
                ncols=100)
    for header in headers:
        info = (
                header["Event"],
                header["Result"],
                int(header["WhiteElo"]) if header["WhiteElo"].isnumeric() else 0,
                int(header["BlackElo"]) if header["BlackElo"].isnumeric() else 0,
                header["TimeControl"],
                header["Termination"]
                )
        queue.put(info)
        # Update progress bar
        update_value = headers.total_num_bytes_read() - pbar.n
        update_value = update_value if pbar.n + update_value <= headers.total_num_bytes() else headers.total_num_bytes() - pbar.n
        pbar.update(update_value)
    # Signal to the consumer that no more records will be produced
    queue.put(None)
    pbar.close()


def record_consumer(scratch_file_path, queue):
    column_types = {
            "Event": "category",
            "Result": "category",
            "WhiteElo": "uint16",
            "BlackElo": "uint16",
            "TimeControl": "category",
            "Termination": "category"
            }
    columns = [
            "Event",
            "Result",
            "WhiteElo",
            "BlackElo",
            "TimeControl",
            "Termination"
            ]
    is_first_write = True
    is_receiving = True
    buffer = list()
    while is_receiving:
        record = queue.get(block=True) # Wait for a record
        is_receiving = record is not None
        if is_receiving and len(buffer) < MAX_BUFFER_LEN:
            buffer.append(record)
            continue
        df = pd.DataFrame(
                data=buffer,
                columns=columns).astype(column_types)
        buffer.clear()
        table = pa.Table.from_pandas(df)
        schema = table.schema
        if is_first_write:
            parquet_writer = pq.ParquetWriter(
                    scratch_file_path,
                    schema,
                    compression="zstd")
            is_first_write = False
        parquet_writer.write_table(table)
    parquet_writer.close() # Commit changes


def process_headers(download_link):
    new_file_name = file_name_from_link(download_link)
    scratch_file_path = f"{SCRATCH_DIR}/{new_file_name}.zstd"
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1).strip()
    queue = Queue(maxsize=MAX_QUEUE_SIZE)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            producer = executor.submit(record_producer, download_link, queue)
            consumer = executor.submit(
                    record_consumer, scratch_file_path, queue)

            # This will wait for threads and raise exceptions
            producer.result()
            consumer.result()
        gDrive = GDrive(CREDENTIALS_JSON)
        gDrive.write_file(scratch_file_path, PARENT_DIR_ID, new_file_name)
    except Exception as e:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Error processing {date}: '{e}'."
    else:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Successfully processed {date}."
    finally:
        if os.path.exists(scratch_file_path):
            os.remove(scratch_file_path)
    return output_message


if __name__ == "__main__":
    freeze_support() # Support for Windows
    tqdm.set_lock(RLock()) # To manage output concurency

    download_links = sorted(
        requests.get(DOWNLOAD_LIST).text.split('\n'),
        reverse=False)

    tqdm.write(f"Found {len(download_links)} files to download.")
    tqdm.write("Checking how many were already processed...")

    gDrive = GDrive(CREDENTIALS_JSON)
    existing_files = gDrive.get_files(PARENT_DIR_ID)
    unprocessed_links = [download_link for download_link in download_links
            if f"{file_name_from_link(download_link)}.zstd" not in existing_files]

    tqdm.write(f"{len(download_links) - len(unprocessed_links)} files already processed.")
    tqdm.write(f"Processing remaining {len(unprocessed_links)} files...")
    tqdm.write(f"Using {N_PROCESSES} processes.")

    tqdm.write(f"Creating the scratch space directory '{SCRATCH_DIR}'...") 
    if not os.path.exists(SCRATCH_DIR):
        os.mkdir(SCRATCH_DIR)

    chunk_size = 1
    lock = tqdm.get_lock()
    try:
        with Pool(N_PROCESSES, initializer=tqdm.set_lock, initargs=(lock,)) as pool:
            results = pool.imap_unordered(
                    process_headers,
                    unprocessed_links,
                    chunksize=chunk_size)
            for result in tqdm(results, desc="Links processed", total=len(unprocessed_links), position=0, ncols=100,
                               leave=True):
                tqdm.write(result, file=sys.stderr)
    finally:
        shutil.rmtree(SCRATCH_DIR)

