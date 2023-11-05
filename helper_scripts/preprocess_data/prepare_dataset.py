# Usage: python3 download_game_data.py
# 
# Iterates through the games in the PGN files and constructs a dataframe from
# them. This dataframe is then stored in Google Drive.
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


INPUT_DATA_DIR = "/Users/luiz/Documents/Projects/Chessformer/data/raw_games"
OUTPUT_DIR = "/Users/luiz/Documents/Projects/Chessformer/data/training_data"


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


def process_file(input_file):
    date = get_date(input_file)
    processed_file_name = get_processed_file_name(input_file)
    input_file_path = f"{INPUT_DATA_DIR}/{input_file}"
    processed_file_path = f"{OUTPUT_DIR}/{processed_file_name}"
    iterator = ParquetIterator(input_file_path, columns=["WhiteElo",
                                                         "BlackElo",
                                                         "Moves",
                                                         "Result"])
    date = get_date(input_file)
    pbar = tqdm(total=iterator.total_num_bytes(),
                unit='B',
                position=1,
                unit_scale=True,
                leave=False,
                desc=date,
                ncols=100)
    try:
        with tf.io.TFRecordWriter(processed_file_path, "ZLIB") as writer:
            for batch in iterator:
                for record in tqdm(batch.to_pylist(), position=2, leave=False):
                    example = convert_record_to_tf_example(record)
                    writer.write(example)
                # Update progress bar
                update_value = iterator.total_num_bytes_read() - pbar.n
                update_value = (update_value
                                if pbar.n + update_value <= iterator.total_num_bytes()
                                else iterator.total_num_bytes() - pbar.n)
                pbar.update(update_value)
    except Exception as e:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Error processing {date}: '{e}'."
    else:
        timestamp = datetime.now()
        output_message = f"{timestamp}: Successfully processed {date}."
    finally:
        pbar.close()
    return output_message


if __name__ == "__main__":
    input_files = os.listdir(INPUT_DATA_DIR)

    tqdm.write(f"Found {len(input_files)} files to process.")
    tqdm.write("Checking how many were already processed...")

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    unprocessed_files = get_unprocessed_files(input_files)
    unprocessed_files.sort(reverse=False)

    tqdm.write(f"{len(input_files) - len(unprocessed_files)} files already processed.")
    tqdm.write(f"Processing remaining {len(unprocessed_files)} files...")

    for file in tqdm(unprocessed_files, desc="Files processed",
                     total=len(unprocessed_files), position=0, ncols=100,
                     leave=True):
        tqdm.write(process_file(file))
