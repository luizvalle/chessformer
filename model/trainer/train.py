import tensorflow as tf
import argparse
import os
import sys
import datetime
import time

from tensorflow.python.client import device_lib
from trainer.data import Dataset
from trainer.model import ChessformerResultClassifier


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


def parse_args():
    parser = argparse.ArgumentParser(
            description="Keras Chess Game Result Predictor")
    parser.add_argument("--training_data_dir", dest="training_data_dir",
                        required=True, type=str,
                        help="Directory with tfrecord files to be used for training.")
    parser.add_argument("--validation_data_dir", dest="validation_data_dir",
                        required=True, type=str,
                        help="Directory with tfrecord files to be used for validation.")
    parser.add_argument("--compression", dest="compression",
                        default="", type=str,
                        help="The compression used by the dataset.")
    parser.add_argument("--batch_size", dest="batch_size",
                        default=128, type=int,
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
    parser.add_argument("--max_checkpoints_to_keep", dest="max_checkpoints_to_keep",
                        default=5, type=int,
                        help="After how many batches to log metric updates.")
    parser.add_argument("--tensorboard_log_dir", dest="tensorboard_log_dir",
                        default=os.getenv("AIP_TENSORBOARD_LOG_DIR"), type=str,
                        help="The location to save the training logs.")
    parser.add_argument("--batch_log_frequency", dest="batch_log_frequency",
                        default=100, type=int,
                        help="After how many batches to print metric updates.")
    parser.add_argument("--epochs", dest="epochs",
                        default=25, type=int,
                        help="Number of epochs.")
    parser.add_argument("--num_encoder_layers", dest="num_encoder_layers",
                        default=2, type=int,
                        help="Number of encoder layers.")
    parser.add_argument("--num_attention_heads", dest="num_attention_heads",
                        default=3, type=int,
                        help="The number of self-attention heads in each encoder layer.")
    parser.add_argument("--embedding_dim", dest="embedding_dim",
                        default=512, type=int,
                        help="The dimension of the emebedding for each token.")
    parser.add_argument("--encoder_feed_forward_dim", dest="encoder_feed_forward_dim",
                        default=512, type=int,
                        help="The size of the hidden layer in the encoder's feed-forward network.")
    parser.add_argument("--dropout_rate", dest="dropout_rate",
                        default=0.1, type=float,
                        help="The dropout rate used in the encoder (in the range [0, 1)).")
    parser.add_argument("--head_feed_forward_dim", dest="head_feed_forward_dim",
                        default=64, type=int,
                        help="The size of the hidden layer in the classification head feed-forward network.")
    parser.add_argument("--max_game_token_length", dest="max_game_token_length",
                        default=1024, type=int,
                        help="The maximum number of tokens games will be trimmed to.")
    args = parser.parse_args()
    return args


@tf.function
def train_step(
        moves, true_results, model, loss_fn, optimizer,
        cumulative_acc_metric, cumulative_loss_metric):
    with tf.GradientTape() as tape:
        predicted_results = model(moves, training=True)
        loss_value = loss_fn(true_results, predicted_results)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    cumulative_acc_metric.update_state(true_results, predicted_results)
    cumulative_loss_metric.update_state(true_results, predicted_results)
    batch_accuracy = tf.keras.metrics.categorical_accuracy(
            true_results, predicted_results)
    return loss_value, tf.math.reduce_mean(batch_accuracy)


@tf.function
def val_step(
        moves, true_results, model, loss_fn, cumulative_acc_metric,
        cumulative_loss_metric):
    predicted_results = model(moves, training=False)
    loss_value = loss_fn(true_results, predicted_results)
    cumulative_acc_metric.update_state(true_results, predicted_results)
    cumulative_loss_metric.update_state(true_results, predicted_results)
    batch_accuracy = tf.keras.metrics.categorical_accuracy(
            true_results, predicted_results)
    return loss_value, tf.math.reduce_mean(batch_accuracy)


def main():
    args = parse_args()

    print("PARAMETERS:")
    for arg in vars(args):
        print(f"\t{arg} = {getattr(args, arg)}")

    dataset = Dataset(
            args.training_data_dir, args.validation_data_dir,
            compression=args.compression,
            max_game_length=args.max_game_token_length)

    train_dataset, val_dataset = dataset.split()
    train_dataset = dataset.make_batches(
            train_dataset, args.batch_size, args.shuffle_buffer_size)
    val_dataset = dataset.make_batches(
            val_dataset, args.batch_size, args.shuffle_buffer_size)

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

    cumulative_acc_metric = tf.keras.metrics.CategoricalAccuracy()
    cumulative_loss_metric = tf.keras.metrics.CategoricalCrossentropy()

    save_logs = args.tensorboard_log_dir is not None
    if save_logs:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = os.path.join(
                args.tensorboard_log_dir, current_time, "train")
        val_log_dir = os.path.join(
                args.tensorboard_log_dir, current_time, "val")
        train_epoch_summary_writer = tf.summary.create_file_writer(
                os.path.join(train_log_dir, "epoch"))
        val_epoch_summary_writer = tf.summary.create_file_writer(
                os.path.join(val_log_dir, "epoch"))

    save_checkpoints = args.model_checkpoint_dir is not None
    if save_checkpoints:
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_manager = tf.train.CheckpointManager(
                checkpoint, directory=args.model_checkpoint_dir,
                max_to_keep=args.max_checkpoints_to_keep)
        loaded_checkpoint_file = checkpoint_manager.restore_or_initialize()
        if loaded_checkpoint_file:
            print(f"Loaded the checkpoint from '{loaded_checkpoint_file}'")
        else:
            print("No checkpoint found.")

    for epoch in range(args.epochs):
        print(f"\nStart of epoch {epoch}")
        start_time = time.time()

        if save_logs:
            train_batch_summary_writer = tf.summary.create_file_writer(
                    os.path.join(train_log_dir, f"batch/epoch/{epoch}"))
            val_batch_summary_writer = tf.summary.create_file_writer(
                    os.path.join(val_log_dir, f"batch/epoch/{epoch}"))

        # Iterate over the batches of the dataset.
        for step, (moves, true_elos, true_results) in enumerate(train_dataset):
            batch_loss, batch_accuracy = train_step(
                    moves, true_results, model, loss_fn, optimizer,
                    cumulative_acc_metric, cumulative_loss_metric)

            if step % args.batch_log_frequency == 0:
                print(
                    f"Training loss (for one batch) at step {step}: {float(batch_loss):.4f}")
                print(f"Seen so far: {(step + 1) * args.batch_size} samples")
                if save_checkpoints:
                    checkpoint_manager.save()

            if save_logs:
                accuracy = cumulative_acc_metric.result()
                loss = cumulative_loss_metric.result()
                with train_batch_summary_writer.as_default():
                    tf.summary.scalar(
                            "cumulative_batch_accuracy", accuracy, step=step)
                    tf.summary.scalar(
                            "cumulative_batch_loss", loss, step=step)
                    tf.summary.scalar(
                            "batch_accuracy", batch_accuracy, step=step)
                    tf.summary.scalar(
                            "batch_loss", batch_loss, step=step)

        # Display metrics at the end of each epoch.
        accuracy = cumulative_acc_metric.result()
        loss = cumulative_loss_metric.result()
        print(f"Training accuracy over epoch: {accuracy:.4f}")
        if save_logs:
            with train_epoch_summary_writer.as_default():
                tf.summary.scalar(
                        "epoch_accuracy", accuracy, step=epoch)
                tf.summary.scalar(
                        "epoch_loss", loss, step=epoch)

        # Reset training metrics at the end of each epoch
        cumulative_acc_metric.reset_states()
        cumulative_loss_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for step, (moves, true_elos, true_results) in enumerate(val_dataset):
            batch_loss, batch_accuracy = val_step(
                    moves, true_results, model, loss_fn, cumulative_acc_metric,
                    cumulative_loss_metric)
            if save_logs:
                accuracy = cumulative_acc_metric.result()
                loss = cumulative_loss_metric.result()
                with val_batch_summary_writer.as_default():
                    tf.summary.scalar(
                            "cumulative_batch_accuracy", accuracy, step=step)
                    tf.summary.scalar(
                            "cumulative_batch_loss", loss, step=step)
                    tf.summary.scalar(
                            "batch_accuracy", batch_accuracy, step=step)
                    tf.summary.scalar(
                            "batch_loss", batch_loss, step=step)

        accuracy = cumulative_acc_metric.result()
        loss = cumulative_loss_metric.result()
        if save_logs:
            with val_epoch_summary_writer.as_default():
                tf.summary.scalar("epoch_accuracy", accuracy, step=epoch)
                tf.summary.scalar("epoch_loss", loss, step=epoch)
        cumulative_acc_metric.reset_states()
        cumulative_loss_metric.reset_states()
        print(f"Time taken: {time.time() - start_time:.2f}s")

    if args.model_save_dir:
        model_save_path = f"{args.model_save_dir}/chessformer_result_classifier.keras"
        model.save(model_save_path)


if __name__ == "__main__":
    main()
