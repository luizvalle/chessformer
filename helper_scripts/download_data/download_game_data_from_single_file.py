import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import re
import requests
import sys
import os
import shutil
import argparse

from chess_iterators import CompressedPgnGameIterator
from gdrive import GDrive
from tqdm.auto import tqdm
from datetime import datetime
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Event


def file_name_from_link(download_link):
    date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1)
    path = f"{date}.parquet.zstd"
    return path


def get_game_date(headers):
    return f"{headers['UTCDate']} {headers['UTCTime']}"


def record_producer(
        download_link, queue, queue_timeout, min_num_tokens, max_num_tokens,
        max_games, error_event):
    try:
        games = CompressedPgnGameIterator(download_link)
        date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1).strip()
        if max_games:
            pbar = tqdm(total=max_games,
                        position=0,
                        leave=True,
                        desc=f"{date} (download)",
                        ncols=100)
        else:
            pbar = tqdm(total=games.total_num_bytes(),
                        unit='B',
                        position=0,
                        unit_scale=True,
                        leave=True,
                        desc=f"{date} (download)",
                        ncols=100)
            max_games = float("inf")
        num_games_processed = 0
        for game in games:
            if game.had_parsing_errors:
                continue
            if (not game.moves
                or len(game.moves) < min_num_tokens
                or len(game.moves) > max_num_tokens):
                continue
            headers = game.headers
            game_date = get_game_date(headers)
            moves = " ".join(game.moves)
            info = (game_date,
                    headers["Result"],
                    int(headers["WhiteElo"]),
                    int(headers["BlackElo"]),
                    headers["TimeControl"],
                    headers["Termination"],
                    len(game.moves),
                    moves)
            if error_event.is_set():
                break
            queue.put(info, block=True, timeout=queue_timeout)
            # Update progress bar
            if max_games:
                update_value = 1
            else:
                update_value = games.total_num_bytes_read() - pbar.n
                update_value = (update_value
                                if pbar.n + update_value <= games.total_num_bytes()
                                else games.total_num_bytes() - pbar.n)
            pbar.update(update_value)
            num_games_processed += 1
            if num_games_processed >= max_games:
                break
    except Exception as e:
        error_event.set()
        raise e
    finally:
        # Signal to the consumer that no more records will be produced
        queue.put(None)
        pbar.close()


def record_consumer(
        scratch_file_path, queue, queue_timeout, max_buffer_length,
        error_event):
    column_types = {
            "Timestamp": "datetime64[us, UTC]",
            "Result": "category",
            "WhiteElo": "uint16",
            "BlackElo": "uint16",
            "TimeControl": "category",
            "Termination": "category",
            "NumTokens": "uint16",
            "Moves": "string"
            }
    columns = [
            "Timestamp",
            "Result",
            "WhiteElo",
            "BlackElo",
            "TimeControl",
            "Termination",
            "NumTokens",
            "Moves"
            ]
    parquet_schema = pa.schema({
        "Timestamp": pa.timestamp("us", "UTC"),
        "Result": pa.dictionary(pa.int16(), pa.string()),
        "WhiteElo": pa.uint16(),
        "BlackElo": pa.uint16(),
        "TimeControl": pa.dictionary(pa.int16(), pa.string()),
        "Termination": pa.dictionary(pa.int16(), pa.string()),
        "NumTokens": pa.uint16(),
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
            record = queue.get(block=True, timeout=queue_timeout)
            is_receiving = record is not None
            if is_receiving and len(buffer) < max_buffer_length:
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


def process_games(
        download_link, max_queue_size, queue_timeout, max_buffer_length,
        min_num_tokens, max_num_tokens, max_games, scratch_dir, gDrive, gdrive_parent_dir_id):
    try:
        new_file_name = file_name_from_link(download_link)
        scratch_file_path = f"{scratch_dir}/{new_file_name}"
        date = re.search(r"(\d{4}-\d{2}).pgn.zst", download_link).group(1).strip()
        queue = Queue(maxsize=max_queue_size)
        error_event = Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            producer = executor.submit(record_producer, download_link, queue,
                                       queue_timeout, min_num_tokens,
                                       max_num_tokens, max_games, error_event)
            consumer = executor.submit(record_consumer, scratch_file_path,
                                       queue, queue_timeout, max_buffer_length,
                                       error_event)
            producer.result()
            consumer.result()
        gDrive.write_file(
                scratch_file_path, gdrive_parent_dir_id, new_file_name)
    except Exception as e:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Error processing {date}: '{e}'."
        raise e
    else:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Successfully processed {date}."
    return output_message


def parse_args():
    parser = argparse.ArgumentParser(
            description="Script to download game data.")
    parser.add_argument("--download_link", dest="download_link",
                        required=True, type=str,
                        help="The link to the download file.")
    parser.add_argument("--gdrive_parent_dir_id", dest="gdrive_parent_dir_id",
                        required=True, type=str,
                        help="The ID for the gDrive directory to place the files under.")
    parser.add_argument("--gdrive_credentials_file", dest="gdrive_credentials_file",
                        default="./credentials.json", type=str,
                        help="The file containing the cached gDrive credentials.")
    parser.add_argument("--is_service_account_credential", dest="is_service_account_credential",
                        default="False", type=str,
                        help="Whether this is a Google service account credential.")
    parser.add_argument("--scratch_dir", dest="scratch_dir",
                        default="./scratch_space", type=str,
                        help="The directory used for scratch space.")
    parser.add_argument("--delete_scratch_files", dest="delete_scratch_files",
                        default="True", type=str,
                        help="Whether this is a Google service account credential.")
    parser.add_argument("--max_queue_size", dest="max_queue_size",
                        default=1e5, type=int,
                        help="The maximum size to use for the producer-consumer queue.")
    parser.add_argument("--max_buffer_length", dest="max_buffer_length",
                        default=1e5, type=int,
                        help="The maximum size to use for the producer-consumer buffer.")
    parser.add_argument("--queue_timeout", dest="queue_timeout",
                        default=15, type=int,
                        help=("The number of seconds the producer or consumer "
                        "will wait for the queue before it times out."))
    parser.add_argument("--min_num_tokens", dest="min_num_tokens",
                        default=0, type=int,
                        help="The minumum number of tokens a game must have.")
    parser.add_argument("--max_num_tokens", dest="max_num_tokens",
                        default=float("inf"), type=int,
                        help="The maximum number of tokens a game can have.")
    parser.add_argument("--max_games_to_save", dest="max_games_to_save",
                        default=None, type=int,
                        help=("The number of seconds the producer or consumer "
                        "will wait for the queue before it times out."))
    args = parser.parse_args()
    args.is_service_account_credential = args.is_service_account_credential == "True"
    args.delete_scratch_files = args.delete_scratch_files == "True"
    return args


def main():
    args = parse_args()

    print("PARAMETERS:")
    for arg in vars(args):
        print(f"\t{arg} = {getattr(args, arg)}")

    if not os.path.exists(args.scratch_dir):
        os.mkdir(args.scratch_dir)
        
    gDrive = GDrive(
            args.gdrive_credentials_file, args.is_service_account_credential)

    try:
        result = process_games(
                args.download_link, args.max_queue_size, args.queue_timeout,
                args.max_buffer_length, args.min_num_tokens, args.max_num_tokens,
                args.max_games_to_save, args.scratch_dir, gDrive,
                args.gdrive_parent_dir_id)
        tqdm.write(result)
    finally:
        if args.delete_scratch_files:
            shutil.rmtree(args.scratch_dir)

if __name__ == "__main__":
    main()
