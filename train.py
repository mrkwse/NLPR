# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import time
from tensorflow.contrib import learn
from classification_cnn import ClassificationCNN
import datetime


embed_dim = 256
filt_szs = [3,4,5]
num_filt = 256 # 256

batch_size = 64
num_epochs = 200

SB2 = False
if SB2:
    import handle_data

    training_data_path="/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_English_SB2.xml"
    evaluation_data_path="/Users/mrkwse/Documents/University/NLPR/OA/Data/EN_LAPT_SB2_TEST.xml.gold"

else:
    import handle_data_SB1 as handle_data

    training_data_path="/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_SB1_v2.xml"
    evaluation_data_path="/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_SB1_v2.xml"

# Load data
text_train, label_train, train_meta = handle_data.load_data(training_data_path)
text_eval, label_eval, eval_meta = handle_data.load_data(evaluation_data_path)

train_label, label_index = handle_data.binary_labels(label_train, return_index=True)
eval_label = handle_data.binary_labels(label_eval, label_list = label_index)




# Vocab
vocab_processor = learn.preprocessing.VocabularyProcessor(train_meta['max_word_count'])

if SB2:
    train_in = []
    eval_in = []

    for review in text_train:
        re_in = np.array(list(vocab_processor.fit_transform(review)))
        train_in.append(re_in)

    train_in = np.array(train_in)

    for review in text_eval:
        re_in = np.array(list(vocab_processor.fit_transform(review)))
        eval_in.append(re_in)

    eval_in = np.array(eval_in)

else:
    train_in = np.array(list(vocab_processor.fit_transform(text_train)))

    eval_in = np.array(list(vocab_processor.fit_transform(text_eval)))




with tf.Graph().as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True
    )
    session = tf.Session(config=session_config)
    with session.as_default():
        cnn = ClassificationCNN(
            # input_count = train_in.shape[0],
            input_dimensions = train_in.shape[1],
            num_classes = [len(train_label[0]), len(train_label[1])], # consider np.array.shape[1]
            vocabulary_size = len(vocab_processor.vocabulary_),
            embedding_size = embed_dim,
            filter_sizes = filt_szs,
            num_filters = num_filt
        )

        global_step = tf.Variable(0, name="global_step", trainable=False)
        #consider tf.train.RMSPropOptimizer or .GradDescentOptimizer
        optimizer = tf.train.AdamOptimizer(1e-4)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Save results in current directory
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "output", timestamp))
        print("Saving to {}\n".format(out_dir))

        # Loss & accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        accuracy_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Training summaries
        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "training")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph_def)

        # Evaulation summaries FIXME
        eval_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        eval_summary_dir = os.path.join(out_dir, "summaries", "evaluation")
        eval_summary_writer = tf.summary.FileWriter(eval_summary_dir, session.graph_def)

        # Checkpoint support
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        # TODO: Store vocabulary?

        session.run(tf.global_variables_initializer())

        def train_step(text_batch, label_batch):
            """
            Single training step
            """
            feed_dict = {
                cnn.inputs: text_batch,  # FIXME: Should input entire set of reviews, not set of sentences in first review
                cnn.labels: label_batch,
                cnn.dropout_keep_prob: 0.1 # best between 0.0 - 0.3
            }

            # FIXME
            _, step, summaries, loss, accuracy = session.run(
                fetches=[train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict=feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def evaluation_step(text_batch, label_batch, writer=None):
            """
            Evaluate model on evaluation data
            """
            feed_dict = {
                cnn.inputs: text_batch,
                cnn.labels: label_batch,
                cnn.dropout_keep_prob: 1.0  # Constant for evaluation
            }

            step, summaries, loss, accuracy = session.run(
                [global_step, eval_summary_op, cnn.loss, cnn.accuracy],
                feed_dict
            )
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)

        # I like trains FIXME
        batches = handle_data.return_batches(
            zip(train_in, train_label),
            batch_size,
            num_epochs
        )

        evaluate_number = 100
        checkpoint_number = 100

        # For each batch in training loop
        for batch in batches:
            text_batch, label_batch = zip(*batch)
            train_step(text_batch, label_batch)
            current_step = tf.train.global_step(session, global_step)
            if current_step % evaluate_number == 0:
                print("\nEvaluation:")
                evaluation_step(eval_in, eval_label, writer=eval_summary_writer)
                print("")
            if current_step % checkpoint_number == 0:
                path = saver.save(session, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
