import os
from absl import logging

PIPELINE_NAME = 'TFRS-movie-ranking'

DATA_ROOT = os.path.join('data', PIPELINE_NAME)
PIPELINE_ROOT = os.path.join('pipelines', PIPELINE_NAME)
METADATA_PATH = os.path.join('pipelines', PIPELINE_NAME, 'metadata.db')
SERVING_MODEL_DIR = os.path.join('serving_model', PIPELINE_NAME)

logging.set_verbosity(logging.INFO)

_trainer_module_file = 'ranking_model_trainer.py'
