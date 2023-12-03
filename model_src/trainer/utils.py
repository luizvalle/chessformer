import tensorflow as tf
import numpy as np

def positional_encoding(length, depth):
    half_depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(half_depth)[np.newaxis, :] / half_depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding[:,:depth], dtype=tf.float32)
