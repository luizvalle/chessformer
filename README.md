# Chessformer

The goal of this project is to predict the result of a chess game and the Elo
ratings of the players involved solely from the sequence of moves. No additional
information, like captures, checks, or checkmate, is provided. Only the piece
that was moved, from which square, and to which square. The architectures of the
models are based on the encoder section of a Transformer. For more details,
including results, see the [report](report/chessformer_report.pdf).

The models are implemented using [TensorFlow](https://www.tensorflow.org/).


## Install the dependencies

The Python packages needed to run the project are listed in the
[requirements.txt](./requirements.txt) file. To install all the packages, run
the following command (assuming the command in run from the root of the
repository):

```sh
pip install -r requirements.txt
```


## Model source code and training script

The model source code can be found in [model_src/trainer/](model_src/trainer).

This directory contains the following files:

* [train.py](model_src/trainer/train.py): The entry point for the
training script. This script allows you to customize the hyper-parameters of the
model and training loop through the command-line arguments passed to the script.
* [layers.py](model_src/trainer/layers.py): Contains the source code for the
model layers.
* [model.py](model_src/trainer/model.py): Implementation of the result
classification and Elo regression models.
* [data.py](model_src/trainer/data.py): Contains utility functions for reading
and preprocessing the training data.
* [utils.py](model_src/trainer/utils.py): Contains helper functions used in the
other modules.

The [model_src/](model_src/) directory contains two files:

* [setup.py](model_src/setup.py): This is needed to setup the package structure
correctly and to package the training code to run on Google Cloud's [Vertex
AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform),
where the models were trained. For more details on the setup.py file, see this
[page](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#setup-py).
* [upload_to_gcloud.sh](model_src/upload_to_gcloud.sh): A Bash script that
packages and uploads the training code to a Google Cloud storage bucket. This is
where the Vertex AI job reads the training code from.

To run the training code locally, run the following commands to setup the
package structure:

```sh
cd model_src/
python setup.py install
```

From `model_src/`, run the following command to start the training script:

```sh
python trainer/train.py \
--training_data_dir=../data/dataset/training \
--validation_data_dir=../data/dataset/validation \
--model_type=result_classifier \
--model_save_dir=/tmp/trained_model \
--model_checkpoint_dir=/tmp/checkpoints \
--tensorboard_log_dir=/tmp/log \
--compression=GZIP \
--batch_size=4 \
--epochs=1 \
--num_encoder_layers=1 \
--num_attention_heads=2 \
--embedding_dim=32 \
--encoder_feed_forward_dim=32 \
--head_feed_forward_dim=16 \
--max_game_token_length=512
```

The script does not print progress to stdout, instead it logs the training
metrics to the `tensorboard_log_dir` directory you provide as
[TensorBoard files](https://www.tensorflow.org/tensorboard/get_started). To see
these metrics in real time, run the following command and point your browser
to the address the command prints out.

```sh
tensorboard --logdir <tensorboard_log_dir>
```

The parameters provided above are arbitrary. For a full list of parameters,
run the following command:

```sh
python trainer/train.py -h
```

The `training_data_dir` and `validation_data_dir` must be directories containing
[TFrecord files](https://www.tensorflow.org/tutorials/load_data/tfrecord). The
entire dataset could not be included in this repository as the files are
too large. However, a toy dataset is included in
[data/dataset](data/dataset) for illustration purposes.


## Notebooks

The notebooks used to analyze the results can be found in
[notebooks/](notebooks/). Note that some cells may not work as the data is not
included in the repository.


## Helper scripts

The [helper_scripts/](helper_scripts/) contains some Python scripts used to
download the training data from the
[Lichess database](https://database.lichess.org/) and then prepare this data
to train with TensorFlow (i.e. convert the data to TFRecord files).

The [helper_scripts/download_data/](helper_scripts/download_data) directory
contains the following files:

* [download_game_metadata.py](helper_scripts/download_data/download_game_metadata.py):
Script to download only the game metadata (i.e. excluding moves) from all files
in the database.
* [download_game_data.py](helper_scripts/download_data/download_game_data.py):
Script to download only the full game data for select types of games from the 
database.
* [download_game_data_from_single_file.py](helper_scripts/download_data/download_game_data_from_single_file.py):
Same as `download_game_data.py` put for just a single file from the database.
database.
* [chess_iterators.py](helper_scripts/download_data/chess_iterators.py): Helper
module that contains the code to iterate through the games in the database.
Used by the download scripts above.
* [gdrive.py](helper_scripts/download_data/gdrive.py): Helper module that
contains the abstractions to save files to Google Drive.

The [helper_scripts/preprocess_data/](helper_scripts/preprocess_data/)
directory contains a script to convert the data downloaded using the scripts
in `download_data/` to TFRecord files that can be used in the training pipeline.

* [prepare_dataset.py](helper_scripts/preprocess_data/prepare_dataset.py): The
entry point for the script.
* [parquet_iterator.py](helper_scripts/preprocess_data/parquet_iterator.py):
Helper module that contains the logic to iterate through a Parquet file. Used
by the main script.
