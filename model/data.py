import tensorflow as tf
import os


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
    def __init__(self, dataset_dir, compression="GZIP", max_game_length=1024):
        files = [f"{dataset_dir}/{file}"
                 for file in os.listdir(dataset_dir)]
        self.dataset = tf.data.TFRecordDataset(filenames=files,
                                               compression_type=compression)
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

    def _make_batches(self, dataset, batch_size, buffer_size):
        return (
                dataset
                .shuffle(buffer_size)
                .batch(batch_size)
                .map(self._prepare_example, tf.data.AUTOTUNE)
                .prefetch(buffer_size=tf.data.AUTOTUNE))

    def get_splits(self, batch_size, buffer_size):
        # TODO: Replace with correct way to split the dataset
        train_dataset = self.dataset.take(1000)
        val_dataset = self.dataset.skip(1000).take(1000)
        return (self._make_batches(train_dataset, batch_size, buffer_size),
                self._make_batches(val_dataset, batch_size, buffer_size))

    def get_vocab_size(self):
        return self.moves_vectorizer.vocabulary_size()
