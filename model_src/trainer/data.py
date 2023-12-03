import tensorflow as tf
import os

from random import shuffle

PIECES = ["p", "n", "b", "r", "q", "k"]
RANK_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8"]
FILE_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h"]
SQUARES = [f + r for r in RANK_NAMES for f in FILE_NAMES]
PROMOTIONS = ["-", "=n", "=b", "=r", "=q"]
POSSIBLE_TOKENS = PIECES + SQUARES + PROMOTIONS

POSSIBLE_RESULTS = ["0-1", "1-0", "1/2-1/2"]

FEATURE_DESCRIPTION = {
    "moves": tf.io.FixedLenFeature([], tf.string, default_value=''),
    "white_elo": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "black_elo": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "result": tf.io.FixedLenFeature([], tf.string, default_value=''),
}


class Dataset():
    def __init__(
            self, train_dataset_dir, validation_dataset_dir, compression,
            max_game_length):
        self.train_files = [os.path.join(train_dataset_dir, file)
                            for file in os.listdir(train_dataset_dir)]
        shuffle(self.train_files)
        self.validation_files = [os.path.join(validation_dataset_dir, file)
                                 for file in os.listdir(validation_dataset_dir)]
        shuffle(self.validation_files)
        self.compression = compression
        self.moves_vectorizer = tf.keras.layers.TextVectorization(
                output_mode="int",
                vocabulary=POSSIBLE_TOKENS,
                standardize=None,
                split="whitespace",
                output_sequence_length=max_game_length)
        self.results_vectorizer = tf.keras.layers.StringLookup(
                vocabulary=POSSIBLE_RESULTS,
                num_oov_indices=1,
                output_mode="one_hot")

    def _prepare_example(self, example_proto):
        example = tf.io.parse_example(example_proto, FEATURE_DESCRIPTION)
        # [:,1:] is used to remove out-of-vocabulary index
        result_embedding = self.results_vectorizer(example["result"])[:,1:]
        tokenized_moves = self.moves_vectorizer(example["moves"])
        white_elo = tf.reshape(example["white_elo"], shape=(-1,1))
        black_elo = tf.reshape(example["black_elo"], shape=(-1,1))
        elos = tf.concat(values=[white_elo, black_elo], axis=1)
        return tokenized_moves, elos, result_embedding

    def split(self):
        train_dataset = tf.data.TFRecordDataset(filenames=self.train_files,
                                                num_parallel_reads=5,
                                                compression_type=self.compression)
        validation_dataset = tf.data.TFRecordDataset(filenames=self.validation_files,
                                                     num_parallel_reads=5,
                                                     compression_type=self.compression)
        return train_dataset, validation_dataset

    def make_batches(self, dataset, batch_size, buffer_size):
        return (
                dataset
                .shuffle(buffer_size)
                .batch(batch_size)
                .map(self._prepare_example, num_parallel_calls=tf.data.AUTOTUNE)
                .prefetch(buffer_size=tf.data.AUTOTUNE))


    def get_vocab_size(self):
        return self.moves_vectorizer.vocabulary_size()
