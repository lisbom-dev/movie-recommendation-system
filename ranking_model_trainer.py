import numpy as np
import tensorflow as tf


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

