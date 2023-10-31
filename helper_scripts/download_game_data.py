# Usage: python3 download_game_data.py
# 
# Iterates through the games in the PGN files and constructs a dataframe from
# them. This dataframe is then stored in Google Drive.
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import re
import requests
import sys
import os
import shutil

from chess_iterators import CompressedPgnGameIterator
from gdrive import GDrive
from tqdm.auto import tqdm
from multiprocessing import Pool, RLock, current_process, set_start_method
from datetime import datetime
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Event


DOWNLOAD_LIST = "https://database.lichess.org/standard/list.txt"
PARENT_DIR_ID = "1PmtuJV7x_tz5sM8SYEE4p5gY8yYyUwov"
CREDENTIALS_JSON = "./credentials.json"  # File with gDrive credentials
IS_SERVICE_ACCOUNT_CREDENTIAL = False
SCRATCH_DIR = "./.games_scratch" # Where temporary data will be stored
N_PROCESSES = 5
MAX_QUEUE_SIZE = 1e4
MAX_BUFFER_LEN = 1e5
QUEUE_TIMEOUT = 60 # seconds
DELETE_SCRATCH_FILES = True
    

def file_name_from_link(download_link):
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1)
    path = f"{date}.parquet.zstd"
    return path


def get_unprocessed_links(download_links):
    gDrive = GDrive(CREDENTIALS_JSON, IS_SERVICE_ACCOUNT_CREDENTIAL)
    existing_files = gDrive.get_files(PARENT_DIR_ID)
    unprocessed_links = [download_link for download_link in download_links
            if f"{file_name_from_link(download_link)}" not in existing_files]
    return unprocessed_links


def is_wanted_game(headers):
    is_blitz = (headers["Event"] == "Rated Blitz game")
    is_normal_termination = (headers["Termination"] == "Normal")
    is_time_forfeit_termination = (headers["Termination"] == "Time forfeit")
    has_white_elo = headers["WhiteElo"].isnumeric()
    has_black_elo = headers["BlackElo"].isnumeric()
    is_variant = ("Variant" in headers)
    return (is_blitz
            and (is_normal_termination or is_time_forfeit_termination)
            and has_white_elo
            and has_black_elo
            and not is_variant)


def get_moves(game):
    board = game.board()
    moves = list()
    for move in game.mainline_moves():
        moves.append(board.san(move))
        board.push(move)
    return " ".join(moves)


def record_producer(download_link, queue, error_event):
    games = CompressedPgnGameIterator(download_link)
    tqdm_pos = current_process()._identity[0] # Index of the current process
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1).strip()
    pbar = tqdm(total=games.total_num_bytes(),
                unit='B',
                position=tqdm_pos,
                unit_scale=True,
                leave=False,
                desc=f"{date} (download)",
                ncols=100)
    try:
        for game in games:
            headers = game.headers
            if not is_wanted_game(headers):
                continue
            if game.errors:
                continue # Game has illegal moves
            if error_event.is_set():
                break
            game_date = (headers["Date"]
                         if headers["Date"]
                         else headers["UTCDate"])
            moves = get_moves(game)
            info = (headers["Event"],
                    game_date,
                    headers["Result"],
                    int(headers["WhiteElo"]),
                    int(headers["BlackElo"]),
                    headers["TimeControl"],
                    headers["Termination"],
                    moves)
            queue.put(info, block=True, timeout=QUEUE_TIMEOUT)
            # Update progress bar
            update_value = games.total_num_bytes_read() - pbar.n
            update_value = (update_value
                            if pbar.n + update_value <= games.total_num_bytes()
                            else games.total_num_bytes() - pbar.n)
            pbar.update(update_value)
    except Exception as e:
        error_event.set()
        raise e
    finally:
        # Signal to the consumer that no more records will be produced
        queue.put(None)
        pbar.close()


def record_consumer(scratch_file_path, queue, error_event):
    column_types = {
            "Event": "category",
            "Date": "datetime64[ns]",
            "Result": "category",
            "WhiteElo": "uint16",
            "BlackElo": "uint16",
            "TimeControl": "category",
            "Termination": "category",
            "Moves": "string"
            }
    columns = [
            "Event",
            "Date",
            "Result",
            "WhiteElo",
            "BlackElo",
            "TimeControl",
            "Termination",
            "Moves"
            ]
    parquet_schema = pa.schema({
        "Event": pa.dictionary(pa.int16(), pa.string()),
        "Date": pa.timestamp(),
        "Result": pa.dictionary(pa.int16(), pa.string()),
        "WhiteElo": pa.uint16(),
        "BlackElo": pa.uint16(),
        "TimeControl": pa.dictionary(pa.int16(), pa.string()),
        "Termination": pa.dictionary(pa.int16(), pa.string()),
        "Moves": pa.string(),
        })
    try:
        parquet_writer = pq.ParquetWriter(
                scratch_file_path,
                parquet_schema,
                compression="zstd")
        is_receiving = True
        buffer = list()
        while is_receiving:
            if error_event.is_set():
                break
            record = queue.get(block=True, timeout=QUEUE_TIMEOUT)
            is_receiving = record is not None
            if is_receiving and len(buffer) < MAX_BUFFER_LEN:
                buffer.append(record)
                continue
            if not buffer:
                break
            df = pd.DataFrame(
                    data=buffer,
                    columns=columns).astype(column_types)
            buffer.clear()
            table = pa.Table.from_pandas(df, schema=parquet_schema)
            parquet_writer.write_table(table)
        if parquet_writer:
            parquet_writer.close() # Commit changes
    except Exception as e:
        error_event.set()
        raise e


def process_headers(download_link):
    new_file_name = file_name_from_link(download_link)
    scratch_file_path = f"{SCRATCH_DIR}/{new_file_name}"
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1).strip()
    queue = Queue(maxsize=MAX_QUEUE_SIZE)
    error_event = Event()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            producer = executor.submit(record_producer, download_link, queue,
                                       error_event)
            consumer = executor.submit(record_consumer, scratch_file_path,
                                       queue, error_event)
            producer.result()
            consumer.result()
        gDrive = GDrive(CREDENTIALS_JSON, IS_SERVICE_ACCOUNT_CREDENTIAL)
        gDrive.write_file(
                scratch_file_path, PARENT_DIR_ID, new_file_name)
    except Exception as e:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Error processing {date}: '{e}'."
    else:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Successfully processed {date}."
    finally:
        if os.path.exists(scratch_file_path) and DELETE_SCRATCH_FILES:
            os.remove(scratch_file_path)
    return output_message


if __name__ == "__main__":
    # The multiprocess library defaults to 'fork' but this is unsafe in macOS
    # as the system may default to threads and thus lead to crashes.
    # See https://github.com/python/cpython/issues/77906
    # Must be the first thing that is called
    set_start_method("spawn")

    tqdm.set_lock(RLock()) # To manage output concurency

    download_links = sorted(
        requests.get(DOWNLOAD_LIST).text.split('\n'),
        reverse=False)

    tqdm.write(f"Found {len(download_links)} files to download.")
    tqdm.write("Checking how many were already processed...")

    unprocessed_links = get_unprocessed_links(download_links)

    tqdm.write(f"{len(download_links) - len(unprocessed_links)} files already processed.")
    tqdm.write(f"Processing remaining {len(unprocessed_links)} files...")
    tqdm.write(f"Using {N_PROCESSES} processes.")

    tqdm.write(f"Creating the scratch space directory '{SCRATCH_DIR}'...") 

    if not os.path.exists(SCRATCH_DIR):
        os.mkdir(SCRATCH_DIR)

    chunk_size = 1
    lock = tqdm.get_lock()
    number_of_processes = min(N_PROCESSES, len(unprocessed_links))
    try:
        with Pool(number_of_processes, initializer=tqdm.set_lock,
                             initargs=(lock,)) as pool:
            results = pool.imap_unordered(process_headers, unprocessed_links,
                    chunksize=chunk_size)
            for result in tqdm(results, desc="Links processed",
                               total=len(unprocessed_links), position=0,
                               ncols=100, leave=True):
                tqdm.write(result)
    finally:
        if DELETE_SCRATCH_FILES:
            shutil.rmtree(SCRATCH_DIR)

