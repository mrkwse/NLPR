# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math

class ClassificationCNN(object):

    def __init__(self, input_dimensions, num_classes,
                 vocabulary_size, embedding_size, num_filters, filter_sizes):

# : [?,?,73,256,1], [3,256,1,82].


        # Placholders for input and labels (None allows for multiple inputs)
        self.inputs = tf.placeholder(tf.int32, shape=[None, input_dimensions], name='input_text')
        self.labels = tf.placeholder(tf.float32, shape=[None, num_classes[0]], name='output_labels')
        #~ Dropout?
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


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
                    filter_shape,
                    stddev=1.0 / math.sqrt(embedding_size), # Possibly fix value§
                    name='weights'
                )
            )

            # biases b
            self.conv_biases = tf.Variable(tf.zeros([num_filters])) # possibly replace with constant
            # self.nce_bias = tf.Variable(tf.constant)

            if i == 1:
                reuse_value = None
            else:
                reuse_value = True
            # Convolution layer
            # conv = tf.layers.conv2d(
            #     inputs = self.embedded_chars_expanded,
            #     filters = num_filters,
            #     kernel_size = [2, 3],   # TODO fix this to something logical FIXME
            #     padding = "VALID",
            #     strides = [1,1],
            #     name = "convolution",
            #     reuse = True,   # May need to set to false
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
        total_filters = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters])

        # Dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        with tf.name_scope("output"):
            #FIXME
            # weights = tf.get_variable(
            #     "self.conv_weights",
            #     tf.truncated_normal([total_filters, num_classes[0]], stddev=0.1), name="c_filter")
            weights = tf.get_variable(
                "self.conv_weights",
                shape = [total_filters, num_classes[0]],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.Variable(tf.constant(0.1, shape=[num_classes[0]]), name="b")
            self.scores = tf.nn.xw_plus_b(self.h_drop, weights, biases, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits = self.scores,
                labels = self.labels
            )
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
