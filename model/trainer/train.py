import tensorflow as tf
import argparse
import os
import sys
import time

from tensorflow.python.client import device_lib
from data import Dataset
from model import ChessformerResultClassifier


# Training loop parameters
SUM_OVER_BATCH_SIZE = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE
MAX_CHECKPOINTS_TO_KEEP = 5
MAX_HOURS_BETWEEN_CHECKPOINTS = 1


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


def parse_args():
    parser = argparse.ArgumentParser(
            description="Keras Chess Game Result Predictor")
    parser.add_argument("--training_data_dir", dest="training_data_dir",
                        required=True, type=str,
                        help="Directory with tfrecord files.")
    parser.add_argument("--train_split", dest="train_split",
                        default=0.8, type=float,
                        help="The proportion of the overall dataset size to use for training.")
    parser.add_argument("--batch_size", dest="batch_size",
                        default=64, type=int,
                        help="The batch size to use in training.")
    parser.add_argument("--shuffle_buffer_size", dest="shuffle_buffer_size",
                        default=20000, type=int,
                        help="The size of the buffer to use when shuffling the dataset.")
    parser.add_argument("--model_save_dir", dest="model_save_dir",
                        default=os.getenv("AIP_MODEL_DIR"), type=str,
                        help="The location to save the final trained model.")
    parser.add_argument("--model_checkpoint_dir", dest="model_checkpoint_dir",
                        default=os.getenv("AIP_CHECKPOINT_DIR"), type=str,
                        help="The location to save the model checkpoints.")
    parser.add_argument("--epochs", dest="epochs",
                        default=25, type=int,
                        help="Number of epochs.")
    parser.add_argument("--num_encoder_layers", dest="num_encoder_layers",
                        default=6, type=int,
                        help="Number of encoder layers.")
    parser.add_argument("--num_attention_heads", dest="num_attention_heads",
                        default=8, type=int,
                        help="The number of self-attention heads in each encoder layer.")
    parser.add_argument("--embedding_dim", dest="embedding_dim",
                        default=512, type=int,
                        help="The dimension of the emebedding for each token.")
    parser.add_argument("--encoder_feed_forward_dim", dest="encoder_feed_forward_dim",
                        default=2048, type=int,
                        help="The size of the hidden layer in the encoder's feed-forward network.")
    parser.add_argument("--dropout_rate", dest="dropout_rate",
                        default=0.1, type=float,
                        help="The dropout rate used in the encoder (in the range [0, 1)).")
    parser.add_argument("--head_feed_forward_dim", dest="head_feed_forward_dim",
                        default=64, type=int,
                        help="The size of the hidden layer in the classification head feed-forward network.")
    args = parser.parse_args()
    return args


@tf.function
def train_step(moves, true_results, model, loss_fn, optimizer, acc_metric):
    with tf.GradientTape() as tape:
        predicted_results = model(moves, training=True)
        loss_value = loss_fn(true_results, predicted_results)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    acc_metric.update_state(true_results, predicted_results)
    return loss_value


@tf.function
def val_step(moves, true_results, model, acc_metric):
    predicted_results = model(moves, training=False)
    acc_metric.update_state(true_results, predicted_results)


def main():
    args = parse_args()

    print("PARAMETERS:")
    for arg in vars(args):
        print(f"\t{arg} = {getattr(args, arg)}")

    dataset = Dataset(args.training_data_dir)

    train_dataset, val_dataset = dataset.split()
    train_dataset = dataset.make_batches(train_dataset, args.batch_size, args.shuffle_buffer_size)
    val_dataset = dataset.make_batches(val_dataset, args.batch_size, args.shuffle_buffer_size)

    vocab_size = dataset.get_vocab_size()

    model = ChessformerResultClassifier(
            num_layers=args.num_encoder_layers,
            vocab_size=vocab_size,
            d_k=args.embedding_dim,
            num_heads=args.num_attention_heads,
            encoder_dff=args.encoder_feed_forward_dim,
            classifier_dff=args.head_feed_forward_dim,
            dropout_rate=args.dropout_rate)

    learning_rate = CustomSchedule(args.embedding_dim)
    optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(
            reduction=SUM_OVER_BATCH_SIZE)

    acc_metric = tf.keras.metrics.CategoricalAccuracy()

    save_checkpoints = args.model_checkpoint_dir is not None
    if save_checkpoints:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, directory=args.model_checkpoint_dir,
                max_to_keep=MAX_CHECKPOINTS_TO_KEEP,
                keep_checkpoint_every_n_hours=MAX_HOURS_BETWEEN_CHECKPOINTS)
        loaded_checkpoint_file = checkpoint_manager.restore_or_initialize()
        if loaded_checkpoint_file:
            print(f"Loaded the checkpoint from '{loaded_checkpoint_file}'")
        else:
            print("No checkpoint found.")

    for epoch in range(args.epochs):
        print(f"\nStart of epoch {epoch}")
        start_time = time.time()

        # Iterate over the batches of the dataset.
        for step, (moves, true_elos, true_results) in enumerate(train_dataset):
            loss_value = train_step(
                    moves, true_results, model, loss_fn, optimizer, acc_metric)

            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    f"Training loss (for one batch) at step {step}: {float(loss_value):.4f}")
                print(f"Seen so far: {(step + 1) * args.batch_size} samples")
                if save_checkpoints:
                    checkpoint_manager.save()

        # Display metrics at the end of each epoch.
        accuracy = acc_metric.result()
        print(f"Training accuracy over epoch: {accuracy:.4f}")

        # Reset training metrics at the end of each epoch
        acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for moves, true_elos, true_results in val_dataset:
            val_step(moves, true_results, model, acc_metric)

        accuracy = acc_metric.result()
        print(f"Validation accuracy over epoch: {accuracy:.4f}")
        acc_metric.reset_states()
        print(f"Time taken: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
