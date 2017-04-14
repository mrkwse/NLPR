# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math

class ClassificationCNN(object):

    def __init__(self, input_dimensions, num_classes, vocabulary_size, embedding_size,
                 num_filters, filter_sizes):

        # Placholders for input and lavels
        self.inputs = tf.placeholder(tf.int32, shape=[input_dimensions], name='input_text')
        self.labels = tf.placeholder(tf.float32, shape=[num_classes[0]], name='output_labels')
        #~ Dropout?


        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )
            self.embedded_chars = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []

        # TODO: Repeat convolution sequnce for every text in review

        # TODO: Two convolution sequnces per text? One for aspect, one for sentiment?
        # Convolution sequence?
        for i, filter_size in enumerate(filter_sizes):

            # filter_size x embedding_size x 1 x num_filters
            # batch x height x width x channels
            filter_shape = [filter_size, embedding_size, 1, num_filters]

            self.conv_weights = tf.Variable(
                tf.truncated_normal(
                    [vocabulary_size, embedding_size],
                    stddev=1.0 / math.sqrt(embedding_size), # Possibly fix value§
                    name='weights'
                )
            )

            # biases b
            self.conv_biases = tf.Variable(tf.zeros([num_filters])) # possibly replace with constant
            # self.nce_bias = tf.Variable(tf.constant)

            # Convolution layer
            # conv = tf.layers.conv2d(
            #     inputs = self.embedded_chars_expanded,
            #     filters = self.conv_weights,
            #     padding = "VALID",
            #     strides = [1,1,1,1],
            #     name = "convolution",
            #     activation=tf.nn.relu
            # )
            conv = tf.nn.conv2d(
                input = self.embedded_chars_expanded,
                filter = self.conv_weights,
                padding = "VALID",
                strides = [1, 1, 1, 1],
                name = "convolution"
            )

            # Maxpool layer
            pool = tf.nn.max_pool(
                value = conv,
                ksize = [1, input_dimensions - filter_size + 1, 1, 1],   # k x ... x ... x ...
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "kpool"
            )

            pooled_outputs.append(pool)

            # TODO: Concatenate results of multiple convolutions

        # Concatenate
        total_filter = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(3, pooled_outputs)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filter])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            weights = tf.Variable(
                "self.conv_weights",
                tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="c_filter")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, weights, biases, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        with tf.name_Scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits = self.scores,
                labels = self.labels
            )
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
