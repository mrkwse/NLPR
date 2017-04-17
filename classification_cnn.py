# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import math

class ClassificationCNN(object):

    def __init__(self, input_dimensions, num_classes, vocabulary_size,
                 embedding_size, num_filters, filter_sizes, batch_size,
                 num_poles=None, num_sp_filters=None, sentiment_filter_sizes=None):


        # Placholders for input and labels (None allows for multiple inputs)
        self.inputs = tf.placeholder(tf.int32, shape=[None, input_dimensions], name='input_text')
        self.labels = tf.placeholder(tf.float32, shape=[None, num_classes], name='output_labels')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.sentiment_inputs = tf.placeholder(tf.int32, shape=[None, input_dimensions + 1], name='sentiment_inputs')
        self.sentiment_labels = tf.placeholder(tf.float32, shape=[None, num_poles], name="sentiment_labels")
        self.threshold = tf.placeholder(tf.float32, shape=[None, num_classes], name='threshold')

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

            # Convolution layer
            conv = tf.nn.conv2d(
                input = self.embedded_chars_expanded,
                filter = self.conv_weights,
                padding = "VALID",
                strides = [1, 1, 1, 1],
                name = "aspect_convolution"
            )

            # Maxpool layer
            pool = tf.nn.max_pool(
                value = conv,
                ksize = [1, input_dimensions - filter_size + 1, 1, 1],   # k x ... x ... x ...
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "aspect_pool"
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
            weights = tf.get_variable(
                "self.conv_weights",
                shape = [total_filters, num_classes],
                initializer = tf.contrib.layers.xavier_initializer()
            )
            biases = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="biases")
            # self.scores = tf.nn.xw_plus_b(self.h_drop, weights, biases, name="scores")
            self.scores = tf.add(tf.matmul(self.h_drop, (weights)), biases)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.relu = tf.nn.relu(self.scores)
            # threshold
            # if self.relu.shape[0] > 0:
            #     thresh = np.full((self.relu.shape[0], num_classes), 5)
            # else:
            #     thresh = np.full((batch_size, num_classes), 5)

            # self.sigscores = tf.floor(self.relu)
            # self.predictions = tf.to_float(tf.greater(self.relu, self.threshold))

            # predict = []
            # for sentence in self.scores:
            #     s_predict = []
            #     for class_val in sentence:
            #         if class_val > 10:
            #             s_predict.append(1)
            #         else:
            #             s_predict.append(0)
            #
            #     predict.append(s_predict)
            #
            # self.predictions = tf.constant(predict)

        # self.predictions should fill
        with tf.name_scope("aspect_loss"):
            self.losses = tf.nn.sigmoid_cross_entropy_with_logits(
                logits = self.scores,
                labels = self.labels,
                name = "sigmoid_cross_entropy_loss"
            )
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope("aspect_accuracy"):
            # correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            correct_predictions = tf.equal(self.scores, self.labels)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")



        # somehow expand inputs
        # self.classified_chars =  tf.concat([self.embedded_chars_expanded, tf.to_float(self.predictions)], 1)

    if 0:
        sent_pools = []

        for i, filter_size in enumerate(sentiment_filter_sizes):

            filter_shape = [filter_size, embedding_size, 1, num_filters]

            self.sp_weights = tf.Variable(
                tf.truncated_normal(
                    filter_shape,
                    stddev=1.0 / math.sqrt(embedding_size),
                    name="sp_weights"
                )
            )

            self.sp_biases = tf.Variable(tf.zeros([num_filters]))

            conv = tf.nn.conv2d(
                input = self.classified_chars,
                filter = self.sp_weights,
                padding = "VALID",
                strides = [1,1,1,1],
                name = "sentiment_convolution"
            )

            # Maxpool layer
            pool = tf.nn.max_pool(
                value = conv,
                ksize = [1, input_dimensions - filter_size + 1, 1, 1],   # k x ... x ... x ...
                strides = [1, 1, 1, 1],
                padding = "VALID",
                name = "aspect_pool"
            )

            sent_pools.append(pool)

        total_filters = num_sp_filters * len(sentiment_filter_sizes)

    if 0:
        with tf.name_scope("sentiment_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits = self.sentiment_scores,
                labels = self.sentiment_labels,
                name = "softmax_cross_entropy_loss"
            )
            self.sentiment_loss = tf.reduce_mean(losses)
