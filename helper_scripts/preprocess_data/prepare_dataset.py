# Usage: python3 download_game_data.py
# 
# Iterates through the games in the PGN files and constructs a dataframe from
# them. This dataframe is then stored in Google Drive.
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import re
import sys
import os
import chess
import tensorflow as tf

from parquet_iterator import ParquetIterator
from tqdm.auto import tqdm
from multiprocessing import Pool, RLock, current_process, set_start_method
from datetime import datetime
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from threading import Event


INPUT_DATA_DIR = "/Users/luiz/Documents/Projects/Chessformer/data/raw_games"
OUTPUT_DIR = "/Users/luiz/Documents/Projects/Chessformer/data/training_data"
N_PROCESSES = 5
MAX_QUEUE_SIZE = 1e6
QUEUE_TIMEOUT = 15 # seconds
    

def get_date(input_file):
    return re.search(r"(\d{4}-\d{2}).parquet.zstd", input_file).group(1)


def get_processed_file_name(input_file):
    date = get_date(input_file)
    path = f"{date}.tfrecord.zlib"
    return path


def get_unprocessed_files(files):
    existing_files = set(os.listdir(OUTPUT_DIR))
    unprocessed_files = [file for file in files
            if get_processed_file_name(file) not in existing_files]
    return unprocessed_files


def standardize_moves(moves):
    board = chess.Board()
    std_moves = list()
    for move in moves.split(" "):
        board_move = board.push_san(move)
        start_square_idx = board_move.from_square
        end_square_idx = board_move.to_square
        start_square = chess.SQUARE_NAMES[start_square_idx]
        end_square = chess.SQUARE_NAMES[end_square_idx]
        if board_move.promotion:
            piece = "p" # Only pawns can promote
            promotion = f"={chess.piece_symbol(board_move.promotion)}"
        else:
            piece = chess.piece_symbol(board.piece_type_at(end_square_idx))
            promotion = "-"
        converted_move = f"{piece} {start_square} {end_square} {promotion}"
        std_moves.append(converted_move)
    return " ".join(std_moves)


def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_example(moves, white_elo, black_elo, result):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
  feature = {
          "moves": bytes_feature(moves),
          "white_elo": int64_feature(white_elo),
          "black_elo": int64_feature(black_elo),
          "result": bytes_feature(result)
          }
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


def convert_record_to_tf_example(record):
    white_elo, black_elo = record["WhiteElo"], record["BlackElo"]
    result = record["Result"].encode("utf-8")
    moves = standardize_moves(record["Moves"]).encode("utf-8")
    example = serialize_example(moves, white_elo, black_elo, result)
    return example


def record_producer(input_file, queue, error_event):
    try:
        input_file_path = f"{INPUT_DATA_DIR}/{input_file}"
        iterator = ParquetIterator(input_file_path, columns=["WhiteElo",
                                                             "BlackElo",
                                                             "Moves",
                                                             "Result"])
        tqdm_pos = current_process()._identity[0] # Index of the current process
        date = get_date(input_file)
        pbar = tqdm(total=iterator.total_num_bytes(),
                    unit='B',
                    position=tqdm_pos,
                    unit_scale=True,
                    leave=False,
                    desc=date,
                    ncols=100)
        for batch in iterator:
            for record in batch.to_pylist():
                queue.put(record, block=True, timeout=QUEUE_TIMEOUT)
            # Update progress bar
            update_value = iterator.total_num_bytes_read() - pbar.n
            update_value = (update_value
                            if pbar.n + update_value <= iterator.total_num_bytes()
                            else iterator.total_num_bytes() - pbar.n)
            pbar.update(update_value)
    except Exception as e:
        error_event.set()
        raise e
    finally:
        # Signal to the consumer that no more records will be produced
        queue.put(None)
        pbar.close()


def record_consumer(processed_file_name, queue, error_event):
    is_receiving = True
    processed_file_path = f"{OUTPUT_DIR}/{processed_file_name}"
    try:
        with tf.io.TFRecordWriter(processed_file_path) as writer:
            while True:
                if error_event.is_set():
                    break
                record = queue.get(block=True, timeout=QUEUE_TIMEOUT)
                is_receiving = record is not None
                if not is_receiving:
                    break
                example = convert_record_to_tf_example(record)
                writer.write(example)
    except Exception as e:
        error_event.set()
        raise e


def process_file(input_file):
    date = get_date(input_file)
    processed_file_name = get_processed_file_name(input_file)
    try:
        queue = Queue(maxsize=MAX_QUEUE_SIZE)
        error_event = Event()
        with ThreadPoolExecutor(max_workers=2) as executor:
            producer = executor.submit(record_producer, input_file, queue,
                                       error_event)
            consumer = executor.submit(record_consumer, processed_file_name,
                                       queue, error_event)
            producer.result()
            consumer.result()
    except Exception as e:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Error processing {date}: '{e}'."
    else:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Successfully processed {date}."
    return output_message


if __name__ == "__main__":
    # The multiprocess library defaults to 'fork' but this is unsafe in macOS
    # as the system may default to threads and thus lead to crashes.
    # See https://github.com/python/cpython/issues/77906
    # Must be the first thing that is called
    set_start_method("spawn")

    tqdm.set_lock(RLock()) # To manage output concurency

    input_files = os.listdir(INPUT_DATA_DIR)

    tqdm.write(f"Found {len(input_files)} files to process.")
    tqdm.write("Checking how many were already processed...")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    unprocessed_files = get_unprocessed_files(input_files)

    tqdm.write(f"{len(input_files) - len(unprocessed_files)} files already processed.")
    tqdm.write(f"Processing remaining {len(unprocessed_files)} files...")
    tqdm.write(f"Using {N_PROCESSES} processes.")

    chunk_size = 1
    lock = tqdm.get_lock()
    number_of_processes = min(N_PROCESSES, len(unprocessed_files))
    with Pool(number_of_processes, initializer=tqdm.set_lock,
                         initargs=(lock,)) as pool:
        results = pool.imap_unordered(process_file, unprocessed_files,
                chunksize=chunk_size)
        for result in tqdm(results, desc="Files processed",
                           total=len(unprocessed_files), position=0,
                           ncols=100, leave=True):
            tqdm.write(result)

