#!/bin/sh
GCLOUD_BUCKET=chessformer-training-code

python setup.py sdist --formats=zip
gcloud storage cp dist/*.zip gs://$GCLOUD_BUCKET
rm -rf dist/ trainer.egg-info/
