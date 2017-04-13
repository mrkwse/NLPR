import tensorflow as tf
import numpy as np
import math

class ClassificationCNN(object):

    def __init__(self, batch_size, num_classes, vocabulary_size, embedding_size,
                 num_filters, filter_sizes):

        # Placholders for input and lavels
        self.inputs = tf.placeholder(tf.int32, shape=[batch_size], name='input_text')
        self.labels = tf.placeholder(tf.float32, shape=[num_classes], name='output_labels')
        #~ Dropout?


        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )
            self.embedded_chars = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            self.enbedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):
            # filter_size x embedding_size x 1 x num_filters
            # batch x height x width x channels
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            c_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1, name="c_filter")
            # bias = tf.Variable(tf.constant)


            conv = tf.layers.conv2d(
                inputs = self.embedded_chars_expanded,
                filters = c_filter,
                padding = "VALID",
                strides = [1,1,1,1],
                name = "convolution"
            )

            pool = tf.nn.max_pool(
                value = conv,
                ksize = [1, sequence_length - filter_size + 1, 1, 1],   # k x ... x ... x ...
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "kpool"
            )

            pooled_outputs.append(pool)
            self.h_pool = tf.concat(3, pooled_outputs)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        num_filters_total = num_filters * len(filter_sizes)

        self.nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size))
        )
        self.nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
