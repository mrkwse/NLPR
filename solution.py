import keras.layers as layers
from keras.models import Model
import keras.callbacks as callbacks
import keras.preprocessing.text as text
import os
import time
import numpy as np
import preprocessing
import prediction_tools as predict

embed_dim = 64 # embedding dimensions
num_filters = 256 # number of convolution filters
aspect_filter_dimensions = [4,5,6] # Topology of convolution layers for aspect
sentiment_filter_dimensions = [4,5] # Same for sentiment
dropout_keep_prob = 0.1 # Probability value for dropout layers

batch_size = 64 # Number of examples per batch processed
num_epochs = 5 # Number of epochs to evaluate over for aspect
num_sentiment_epochs = 7 # same for sentiment

training_data_path = os.environ['TRAIN_PATH'] #/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_SB1_v2.xml
evaluation_data_path = os.environ['EVAL_PATH'] #/Users/mrkwse/Documents/University/NLPR/OA/Data/EN_LAPT_SB1_TEST_.xml.gold

print('Loading data...')
text_train, label_train, train_meta = preprocessing.load_data(training_data_path)
text_eval, label_eval, eval_meta = preprocessing.load_data(evaluation_data_path)

print('Building vocabulary...')
# Build vocabulary from combined text
vocabulary = preprocessing.vocabulary_compile(text_train + text_eval)
vocab_size = len(vocabulary)
print('Vocabulary size: ' + str(vocab_size))


# Convert string aspect labels to boolean labels
train_label, label_index = preprocessing.boolean_labels(label_train, return_index=True)
eval_label = preprocessing.boolean_labels(label_eval, label_list=label_index)

# Build vector representation of text inputs using vocabulary
training_inputs = preprocessing.convert_text(text_train, vocabulary, train_meta)
evaluation_inputs = preprocessing.convert_text(text_eval, vocabulary, train_meta)

training_labels_boolean = preprocessing.boolean_combined(label_train, label_index)
train_boolean_sentiment, train_boolean_sentiment_expanded, train_int_pairs, train_int_pairs_expanded = preprocessing.boolean_to_int(training_labels_boolean)

evaluation_labels_boolean = preprocessing.boolean_combined(label_eval, label_index)
eval_boolean_sentiment, eval_boolean_sentiment_expanded, eval_int_pairs, eval_int_pairs_expanded = preprocessing.boolean_to_int(evaluation_labels_boolean)

# Clear unused variables (remained incase of topology changes)
del train_boolean_sentiment, eval_boolean_sentiment
del train_int_pairs_expanded, eval_int_pairs_expanded

print('Preprocessing complete.')

aspect_inputs = layers.Input(shape=(training_inputs.shape[1],), name="aspect_input")

embedded_chars = layers.Embedding(
                    input_dim = vocab_size,
                    output_dim = embed_dim,
                    input_length = training_inputs.shape[1]
                 )(aspect_inputs)

reshape = layers.core.Reshape(
                target_shape = (training_inputs.shape[1],embed_dim,1)
          )(embedded_chars)

pooled_aspect_outputs = []

for i, filter_size in enumerate(aspect_filter_dimensions):

    aspect_conv = layers.convolutional.Conv2D(
        filters = num_filters,
        kernel_size = (filter_size, embed_dim),
        strides = 1,
        padding = 'valid',
        name = "aspect_conv_" + str(i),
        activation = 'tanh'
    )(reshape)

    aspect_pool = layers.pooling.MaxPooling2D(
        pool_size = (training_inputs.shape[1] - filter_size + 1, 1),
        strides = 1,
        padding = 'valid',
        name = 'aspect_pool_' + str(i),
        data_format = "channels_last"
    )(aspect_conv)

    pooled_aspect_outputs.append(aspect_pool)

combined_pool = layers.Concatenate(axis=1)(pooled_aspect_outputs)
flat_aspect = layers.Flatten()(combined_pool)

aspect_drop = layers.Dropout(rate=dropout_keep_prob, name="dropout")(flat_aspect)

aspect_output = layers.Dense(
                    train_label.shape[1],
                    activation='sigmoid',
                    name='main_output'
                )(aspect_drop)

aspect_model = Model(inputs=aspect_inputs, outputs=aspect_output)

print('Compiling aspect model...')
aspect_model.compile(
    optimizer = 'Adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)


# Configure logging for TensorBoard
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "output", timestamp))
print("Saving to {}".format(out_dir))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "aspect/aspect_model")

checkpoints = callbacks.ModelCheckpoint(
    checkpoint_prefix + 'weights.{epoch:02d}.hdf5',
    monitor='val_loss,val_acc',
    verbose=0,
    save_best_only=False,
    mode='auto'
)

board = callbacks.TensorBoard(log_dir=out_dir)

print('Training aspect model...')
aspect_model.fit(
    training_inputs,
    train_label,
    epochs = num_epochs,
    batch_size = batch_size,
    callbacks=[checkpoints, board],
    validation_data=(evaluation_inputs, eval_label)
)

train_predictions = aspect_model.predict(
                        training_inputs,
                        batch_size = batch_size
                    )
evaluation_predictions = aspect_model.predict(
                            evaluation_inputs,
                            batch_size = batch_size
                         )


true_treshold = 0.6

# Predict aspect classes for each input setence, for both training and
# evauluation datasets.
train_classes_i = predict.predict_classes(train_predictions, true_treshold)
eval_classes_i = predict.predict_classes(evaluation_predictions, true_treshold)

# Get dimensions of actual aspects for sentences (i.e., how many opinion
# elements for each sentence)
training_aspect_counts = preprocessing.get_aspect_counts(train_int_pairs)
evaluation_aspect_counts = preprocessing.get_aspect_counts(eval_int_pairs)

# Create a distinct vector input for each (real) opinion for each sentence
# and populate with predicted aspects
predicted_aspect_train_in = predict.add_class_to_text(train_classes_i,
                                                      training_inputs,
                                                      training_aspect_counts)
predicted_aspect_eval_in = predict.add_class_to_text(eval_classes_i,
                                                     evaluation_inputs,
                                                     evaluation_aspect_counts)

# Isolate sentiment classes from training/eval labels
sentiment_training_output = preprocessing.isolate_boolean_sentiment(train_boolean_sentiment_expanded)
sentiment_evaluation_output = preprocessing.isolate_boolean_sentiment(eval_boolean_sentiment_expanded)

sentiment_inputs = layers.Input(
                        shape=(training_inputs.shape[1] + 1,),
                        name="sentiment_input"
                   )


embedded_sent = layers.Embedding(
                    input_dim = vocab_size,
                    output_dim = embed_dim,
                    input_length = predicted_aspect_train_in.shape[1]
                 )(sentiment_inputs)

reshaped_sent = layers.core.Reshape(
                    target_shape = (
                        predicted_aspect_train_in.shape[1],
                        embed_dim,
                        1
                    )
                )(embedded_sent)

sentiment_total_pooled = []

# Allows for multiple convolutions of differing dimensions
for i, filter_size in enumerate(sentiment_filter_dimensions):

    sentiment_conv = layers.convolutional.Conv2D(
        filters = num_filters,
        kernel_size = (filter_size,embed_dim),
        strides = 1,
        padding = 'valid',
        name = "sentiment_conv_" + str(i),
        activation = 'softplus'
    )(reshaped_sent)


    sentiment_pool = layers.pooling.MaxPooling2D(
        pool_size = (predicted_aspect_train_in.shape[1] - filter_size + 1, 1),
        strides = 1,
        padding = 'valid',
        name = 'sentiment_pool_' + str(i),
        data_format = "channels_last"
    )(sentiment_conv)

    sentiment_total_pooled.append(sentiment_pool)

combined_sentiment_pool = layers.Concatenate(axis=1)(sentiment_total_pooled)
flat_sentiment = layers.Flatten()(combined_sentiment_pool)

sentiment_drop = layers.Dropout(
                    rate=dropout_keep_prob,
                    name="dropout"
                 )(flat_sentiment)

sentiment_output = layers.Dense(
                        sentiment_training_output.shape[1],
                        activation='sigmoid',
                        name='main_output'
                   )(sentiment_drop)


sentiment_model = Model(inputs=sentiment_inputs, outputs=sentiment_output)


print('Compiling sentiment model...')
sentiment_model.compile(
    optimizer = 'Adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

sentiment_checkpoint_prefix = os.path.join(checkpoint_dir, "sentiment/sentiment_model")

sentiment_checkpoints = callbacks.ModelCheckpoint(
        sentiment_checkpoint_prefix + 'weights.{epoch:02d}.hdf5',
        monitor='val_loss,val_acc',
        verbose=0,
        save_best_only=False,
        mode='auto'
)

sentiment_board = callbacks.TensorBoard(log_dir=out_dir)

print('Training sentiment model...')
sentiment_model.fit(
    predicted_aspect_train_in,
    sentiment_training_output,
    epochs = num_sentiment_epochs,
    batch_size = batch_size,
    callbacks=[sentiment_checkpoints, sentiment_board],
    validation_data=(predicted_aspect_eval_in, sentiment_evaluation_output)
)
