import tensorflow as tf
import argparse
import os
import sys
import time

from tensorflow.python.client import device_lib
from data import Dataset
from model import ChessformerResultClassifier


# Dataset parameters
BUFFER_SIZE = 20000
BATCH_SIZE = 128

# Model parameters
NUM_ENCODER_LAYERS = 6
EMBEDDING_DIM = 512
NUM_ATTENTION_HEADS = 8
FEED_FORWARD_DIMENSION = 2048
DROPOUT_RATE = 0.1

# Training loop parameters
SUM_OVER_BATCH_SIZE = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_k, warmup_steps=4000):
        super().__init__()
        self.d_k = tf.cast(d_k, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_k) * tf.math.minimum(arg1, arg2)


@tf.function
def train_step(moves, true_results):
    with tf.GradientTape() as tape:
        predicted_results = model(moves, training=True)
        loss_value = loss_fn(true_results, predicted_results)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    acc_metric.update_state(true_results, predicted_results)
    return loss_value


@tf.function
def val_step(moves, true_results):
    predicted_results = model(moves, training=False)
    acc_metric.update_state(true_results, predicted_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data_dir", dest="training_data_dir",
                        required=True, type=str,
                        help="Directory with tfrecord files.")
    parser.add_argument("--epochs", dest="epochs",
                        default=10, type=int,
                        help="Number of epochs.")
    args = parser.parse_args()
    
    dataset = Dataset(args.training_data_dir)

    train_dataset, val_dataset = dataset.split()
    train_dataset = dataset.make_batches(train_dataset, BATCH_SIZE, BUFFER_SIZE)
    val_dataset = dataset.make_batches(val_dataset, BATCH_SIZE, BUFFER_SIZE)

    vocab_size = dataset.get_vocab_size()

    model = ChessformerResultClassifier(
            NUM_ENCODER_LAYERS, vocab_size, EMBEDDING_DIM, NUM_ATTENTION_HEADS,
            FEED_FORWARD_DIMENSION, DROPOUT_RATE)

    learning_rate = CustomSchedule(EMBEDDING_DIM)
    optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
            reduction=SUM_OVER_BATCH_SIZE)

    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(args.epochs):
        print(f"\nStart of epoch {epoch}")
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (moves, true_elos, true_results) in enumerate(train_dataset):
            loss_value = train_step(moves, true_results)

            # TODO: Change this
            # Log every 1 batch.
            if step % 1 == 0:
                print(
                    f"Training loss (for one batch) at step {step}: {float(loss_value):.4f}")
                print(f"Seen so far: {(step + 1) * BATCH_SIZE} samples")

        # Display metrics at the end of each epoch.
        accuracy = acc_metric.result()
        print(f"Training accuracy over epoch: {accuracy:.4f}")

        # Reset training metrics at the end of each epoch
        acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for moves, true_elos, true_results in val_dataset:
            val_step(moves, true_results)

        accuracy = acc_metric.result()
        print(f"Validation accuracy over epoch: {accuracy:.4f}")
        val_elo_error_metric.reset_states()
        val_result_acc_metric.reset_states()
        print(f"Time taken: {time.time() - start_time:.2f}s")
