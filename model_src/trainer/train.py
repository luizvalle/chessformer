import tensorflow as tf
import argparse
import os
import sys
import datetime
import time

from trainer.data import Dataset
from trainer.model import ChessformerResultClassifier, ChessformerEloRegressor, ChessformerEloRegressorV2


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
    parser.add_argument("--model_type", dest="model_type",
                        default="result_classifier", type=str,
                        help="Either 'result_classifier' or 'elo_regressor'.")
    parser.add_argument("--pretrained_model_path", dest="pretrained_model_path",
                        default=None, type=str,
                        help="The path to the pretrained .keras model with an encoder layer to be re-used.")
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
    parser.add_argument("--checkpoint_step_frequency", dest="checkpoint_step_frequency",
                        default=100, type=int,
                        help="After how many steps to save checkpoints.")
    parser.add_argument("--epochs", dest="epochs",
                        default=1, type=int,
                        help="Number of epochs.")
    parser.add_argument("--learning_rate_warmup_steps", dest="learning_rate_warmup_steps",
                        default=4000, type=int,
                        help="Number of training steps before the learning rate starts to decrease.")
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
        moves, true_labels, model, loss_fn, optimizer, cumulative_metrics,
        additional_batch_metrics):
    with tf.GradientTape() as tape:
        predicted_labels = model(moves, training=True)
        loss_value = loss_fn(true_labels, predicted_labels)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    for metric in cumulative_metrics.values():
        metric.update_state(true_labels, predicted_labels)
    batch_metrics = {"batch_loss": loss_value}
    for metric_name, metric in additional_batch_metrics.items():
        batch_metrics[metric_name] = metric(true_labels, predicted_labels)
    return batch_metrics


@tf.function
def val_step(
        moves, true_labels, model, loss_fn, cumulative_metrics,
        additional_batch_metrics):
    predicted_labels = model(moves, training=False)
    loss_value = loss_fn(true_labels, predicted_labels)
    for metric in cumulative_metrics.values():
        metric.update_state(true_labels, predicted_labels)
    batch_metrics = {"batch_loss": loss_value}
    for metric_name, metric in additional_batch_metrics.items():
        batch_metrics[metric_name] = metric(true_labels, predicted_labels)
    return batch_metrics


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

    if args.model_type == "result_classifier":
        model = ChessformerResultClassifier(
                num_layers=args.num_encoder_layers,
                vocab_size=vocab_size,
                d_k=args.embedding_dim,
                num_heads=args.num_attention_heads,
                encoder_dff=args.encoder_feed_forward_dim,
                classifier_dff=args.head_feed_forward_dim,
                dropout_rate=args.dropout_rate)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(
                reduction=SUM_OVER_BATCH_SIZE)
        cumulative_metrics = {
                "accuracy": tf.keras.metrics.CategoricalAccuracy(),
                "loss": tf.keras.metrics.CategoricalAccuracy(),
                }
        additional_batch_metrics = {
                "batch_accuracy": lambda true_labels, predicted_labels:
                tf.math.reduce_mean(tf.keras.metrics.categorical_accuracy(
                    true_labels, predicted_labels))
                }
    elif args.model_type == "elo_regressor":
        model = ChessformerEloRegressor(
                num_layers=args.num_encoder_layers,
                vocab_size=vocab_size,
                d_k=args.embedding_dim,
                num_heads=args.num_attention_heads,
                encoder_dff=args.encoder_feed_forward_dim,
                regressor_dff=args.head_feed_forward_dim,
                dropout_rate=args.dropout_rate)
        loss_fn = tf.keras.losses.MeanSquaredError(
                reduction=SUM_OVER_BATCH_SIZE)
        cumulative_metrics = {
                "batch_loss": tf.keras.metrics.MeanSquaredError(),
                }
        # No metric besides loss, which is already computed, is needed
        additional_batch_metrics = dict()
    elif args.model_type == "elo_regressor_v2":
        model = ChessformerEloRegressorV2(
                num_layers=args.num_encoder_layers,
                vocab_size=vocab_size,
                d_k=args.embedding_dim,
                num_heads=args.num_attention_heads,
                encoder_dff=args.encoder_feed_forward_dim,
                regressor_dff=args.head_feed_forward_dim,
                dropout_rate=args.dropout_rate)
        loss_fn = tf.keras.losses.MeanSquaredError(
                reduction=SUM_OVER_BATCH_SIZE)
        cumulative_metrics = {
                "cumulative_batch_loss": tf.keras.metrics.MeanSquaredError(),
                }
        # No metric besides loss, which is already computed, is needed
        additional_batch_metrics = dict()
    else:
        raise NotImplementedError(
                "model_type can only be 'result_classifer' or 'elo_regressor'.")

    if args.pretrained_model_path:
        # Have to build the model before loading weights
        input_dims = (args.batch_size, args.max_game_token_length)
        model.build(input_dims)
        trained_model = tf.keras.models.load_model(
                args.pretrained_model_path)
        # Assuming the first layer is the encoder
        pretrained_encoder = trained_model.get_layer(index=0)
        current_encoder = model.get_layer(index=0)
        current_encoder.set_weights(pretrained_encoder.get_weights())
        print(f"Loaded encoder from '{args.pretrained_model_path}'.")

    learning_rate = CustomSchedule(
            args.embedding_dim, warmup_steps=args.learning_rate_warmup_steps)
    optimizer = tf.keras.optimizers.Adam(
            learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)


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
        start_time = time.time()

        if save_logs:
            train_batch_summary_writer = tf.summary.create_file_writer(
                    os.path.join(train_log_dir, f"batch/epoch/{epoch}"))
            val_batch_summary_writer = tf.summary.create_file_writer(
                    os.path.join(val_log_dir, f"batch/epoch/{epoch}"))

        # Iterate over the batches of the dataset.
        for step, (moves, true_elos, true_results) in enumerate(train_dataset):
            if args.model_type == "result_classifier":
                true_labels = true_results
            elif args.model_type in {"elo_regressor", "elo_regressor_v2"}:
                true_labels = true_elos
            batch_metrics = train_step(
                    moves, true_labels, model, loss_fn, optimizer,
                    cumulative_metrics, additional_batch_metrics)

            if step % args.checkpoint_step_frequency == 0 and save_checkpoints:
                checkpoint_manager.save()

            if save_logs:
                with train_batch_summary_writer.as_default():
                    for metric_name, metric in cumulative_metrics.items():
                        tf.summary.scalar(
                                f"cumulative_batch_{metric_name}",
                                metric.result(),
                                step=step)
                    for metric_name, metric_value in batch_metrics.items():
                        tf.summary.scalar(
                                metric_name, metric_value, step=step)

        # Save metrics at the end of each epoch.
        if save_logs:
            with train_epoch_summary_writer.as_default():
                for metric_name, metric in cumulative_metrics.items():
                    tf.summary.scalar(
                            f"epoch_{metric_name}", metric.result(),
                            step=epoch)

        # Reset training metrics at the end of each epoch
        for metric in cumulative_metrics.values():
            metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for step, (moves, true_elos, true_results) in enumerate(val_dataset):
            if args.model_type == "result_classifier":
                true_labels = true_results
            elif args.model_type in {"elo_regressor", "elo_regressor_v2"}:
                true_labels = true_elos
            batch_metrics = val_step(
                    moves, true_labels, model, loss_fn, cumulative_metrics,
                    additional_batch_metrics)
            if save_logs:
                with val_batch_summary_writer.as_default():
                    for metric_name, metric in cumulative_metrics.items():
                        tf.summary.scalar(
                                f"cumulative_batch_{metric_name}",
                                metric.result(),
                                step=step)
                    for metric_name, metric_value in batch_metrics.items():
                        tf.summary.scalar(
                                metric_name, metric_value, step=step)

        if save_logs:
            with val_epoch_summary_writer.as_default():
                for metric_name, metric in cumulative_metrics.items():
                    tf.summary.scalar(
                            f"epoch_{metric_name}", metric.result(),
                            step=epoch)

        for metric in cumulative_metrics.values():
            metric.reset_states()
        print(f"Time taken: {time.time() - start_time:.2f}s")

    if args.model_save_dir:
        exported_model_dir = f"{args.model_save_dir}/exported_model"
        saved_model_dir = f"{args.model_save_dir}/saved_model"
        weights_dir = f"{args.model_save_dir}/weights"
        os.makedirs(exported_model_dir, exist_ok=True)
        os.makedirs(saved_model_dir, exist_ok=True)
        os.makedirs(weights_dir, exist_ok=True)
        # Can be used for inference only, contains the forward pass
        model.export(exported_model_dir)
        # Types below can be used for fine-tuning
        tf_save_path = f"{saved_model_dir}/chessformer_result_classifier.tf"
        try:
            model.save(tf_save_path, save_format="tf")
        except Exception as e:
            pass
        keras_save_path = f"{saved_model_dir}/chessformer_result_classifier.keras"
        try:
            model.save(keras_save_path)
        except Exception as e:
            pass
        weights_save_path = f"{weights_dir}/chessformer_result_classifier.weights.h5"
        try:
            model.save_weights(weights_save_path)
        except Exception as e:
            pass


if __name__ == "__main__":
    main()
