import pandas as pd
import numpy as np
import re
import sys
import os
import chess
import tensorflow as tf

from parquet_iterator import ParquetIterator
from tqdm.auto import tqdm
from datetime import datetime


INPUT_DATA_DIR = "/Users/luiz/Documents/Projects/Chessformer/data/raw_game_data"
OUTPUT_DIR = "/Users/luiz/Documents/Projects/Chessformer/data/training_data"
OUT_FILE_SIZE_LIMIT = 200 # MB


def get_date(input_file):
    return re.search(r"(\d{4}-\d{2}).parquet.zstd", input_file).group(1)


def log(message):
    timestamp = datetime.now()
    output_message = f"{timestamp}: {message}"
    tqdm.write(output_message)


def get_unprocessed_files(files):
    existing_files = set(os.listdir(OUTPUT_DIR))
    unprocessed_files = [file for file in files
            if get_processed_file_name(file) not in existing_files]
    return unprocessed_files


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
    moves = record["Moves"].encode("utf-8")
    example = serialize_example(moves, white_elo, black_elo, result)
    return example


if __name__ == "__main__":
    input_files = os.listdir(INPUT_DATA_DIR)
    input_files.sort()

    tqdm.write(f"Found {len(input_files)} files to process.")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    shard = 0
    total_size = 0
    num_games = 0
    processed_file_path = f"{OUTPUT_DIR}/data-{shard}.tfrecord.gzip"
    writer = tf.io.TFRecordWriter(processed_file_path, "GZIP") 
    for file in tqdm(input_files, desc="Files processed",
                     total=len(input_files), position=0, ncols=100,
                     leave=True):
        pbar = None
        try:
            input_file_path = f"{INPUT_DATA_DIR}/{file}"
            iterator = ParquetIterator(input_file_path, columns=["WhiteElo",
                                                                 "BlackElo",
                                                                 "Moves",
                                                                 "Result"]) 
            date = get_date(file)
            pbar = tqdm(total=iterator.total_num_bytes(),
                        unit='B',
                        position=1,
                        unit_scale=True,
                        leave=False,
                        desc=date,
                        ncols=100)
            for i, batch in enumerate(iterator):
                for record in tqdm(batch.to_pylist(),
                                   desc=f"Batch {i + 1}",
                                   position=2,
                                   ncols=100,
                                   leave=False):
                    example = convert_record_to_tf_example(record)
                    size = sys.getsizeof(example) / 1e6
                    if total_size + size > OUT_FILE_SIZE_LIMIT:
                        log(f"Shard {shard} reached {total_size:.2f} MB / {OUT_FILE_SIZE_LIMIT} MB. Games in file: {num_games}.")
                        log("Creating a new shard.")
                        writer.close()
                        shard += 1
                        total_size = 0
                        num_games = 0
                        processed_file_path = f"{OUTPUT_DIR}/data-{shard}.tfrecord.gzip"
                        writer = tf.io.TFRecordWriter(processed_file_path,
                                                      "GZIP") 
                    writer.write(example)
                    total_size += size
                    num_games += 1
                # Update progress bar
                update_value = iterator.total_num_bytes_read() - pbar.n
                update_value = (update_value
                                if pbar.n + update_value <= iterator.total_num_bytes()
                                else iterator.total_num_bytes() - pbar.n)
                pbar.update(update_value)
        except Exception as e:
            log(f"Error processing {date}: '{e}'.")
        else:
            log(f"Successfully processed {date}.")
        finally:
            if pbar:
                pbar.close()
