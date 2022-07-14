from typing import Dict, Text, List

import numpy as np

from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_transform.tf_metadata import schema_utils
import tensorflow as tf
import tensorflow_recommenders as tfrs

from tfx import v1 as tfx
from tfx_bsl.public import tfxio

_FEATURE_KEYS = ['userId', 'movieId']
_LABEL_KEY = 'rating'

_FEATURE_SPEC = {
    'userId': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'movieId': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
    'rating': tf.io.FixedLenFeature(shape=[1], dtype=tf.int64),
}


class MovieRankingModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        embedding_dimension = 32

        user_range = 943
        movie_range = 1682

        unique_user_ids = np.array(range(user_range)).astype(str)
        unique_movie_ids = np.array(range(movie_range)).astype(str)

        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), name='userId', dtype=tf.int64),
            tf.keras.layers.Lambda(lambda x: tf.as_string(x)),
            tf.keras.layers.StringLookup(
                vocabulary=unique_user_ids, mask_token=None
            ),
            tf.keras.layers.Embedding(
                len(unique_user_ids)+1, embedding_dimension
            ),
        ])

        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1,), name='movieId', dtype=tf.int64),
            tf.keras.layers.Lambda(lambda x: tf.as_string(x)),
            tf.keras.layers.StringLookup(
                vocabulary=unique_movie_ids, mask_token=None
            ),
            tf.keras.layers.Embedding(
                len(unique_movie_ids)+1, embedding_dimension
            )
        ])

        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):

        user_id, movie_id = inputs

        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_id)

        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=2))


class MovielensModel(tfrs.models.Model):

    def __init__(self):
        super().__init__()

        self.ranking_model: tf.keras.Model = MovieRankingModel()

        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, inputs: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model((inputs['userId'], inputs['movieId']))

    def compute_loss(self,
                     inputs: Dict[Text, tf.Tensor],
                     training: bool = False) -> tf.Tensor:

        labels = inputs[1]
        rating_predictions = self(inputs[0])

        return self.task(labels=labels, predictions=rating_predictions)


def _input_fn(file_pattern: List[str],
              data_accessor: tfx.components.DataAccessor,
              schema: schema_pb2.Schema,
              batch_size: int = 256) -> tf.data.Dataset:

    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size, label_key=_LABEL_KEY
            ),
        schema
        ).repeat()

def _build_keras_model() -> tf.keras.Model:
    return MovielensModel()


def run_fn(fn_args: tfx.components.FnArgs):

    schema = schema_utils.schema_from_feature_spec(_FEATURE_SPEC)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor, schema, batch_size=8192
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, schema, batch_size=4096
    )

    model = _build_keras_model()

    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        epochs=3,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps
    )

    model.save(fn_args.serving_model_dir)
