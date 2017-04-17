import keras.layers as layers
from keras.models import Model
import keras.callbacks as callbacks
import keras.preprocessing.text as text
import os
import time
import numpy as np
import preprocessing


embed_dim = 50
num_filters = 256
aspect_filter_dimensions = [2,3,4]

batch_size = 64
num_epochs = 50
dropout_keep_prob = 0.1

training_data_path = os.environ['TRAIN_PATH'] #/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_SB1_v2.xml
evaluation_data_path = os.environ['EVAL_PATH'] #/Users/mrkwse/Documents/University/NLPR/OA/Data/EN_LAPT_SB1_TEST_.xml.gold

print('Loading data...')
text_train, label_train, train_meta = preprocessing.load_data(training_data_path)
text_eval, label_eval, eval_meta = preprocessing.load_data(evaluation_data_path)

combined_text = text_train + text_eval

print('Building vocabulary...')
vocabulary, vocabulary_inv = preprocessing.vocabulary_transform(combined_text)
print('Vocabulary size: ' + str(len(vocabulary)))
vocab_size = len(vocabulary) + 1



train_label, label_index = preprocessing.binary_labels(label_train, return_index=True)
eval_label = preprocessing.binary_labels(label_eval, label_list = label_index)


training_inputs = preprocessing.build_input_data(text_train, vocabulary, train_meta)
evaluation_inputs = preprocessing.build_input_data(text_eval, vocabulary, train_meta)

print(training_inputs.shape)
print('Preprocessing complete.')


# This shouldn't be hard coded
inputs = layers.Input(shape=(training_inputs.shape[1],))

embedded_chars = layers.Embedding(
                    input_dim = vocab_size,
                    output_dim = embed_dim,
                    input_length = training_inputs.shape[1]
                 )(inputs)
reshape = layers.core.Reshape((training_inputs.shape[1],embed_dim,1))(embedded_chars)

# TODO[deletecomment] Possibly for loop through multiple filter shapes

pooled_aspect_outputs = []

for i, filter_size in enumerate(aspect_filter_dimensions):
    # filter_shape = [filter_size, embed_dim, 1, num_filters]

    aspect_conv = layers.convolutional.Conv2D(
        filters = num_filters,
        kernel_size = (filter_size,embed_dim),
        strides = (1,1),
        padding = 'valid',
        name = "aspect_conv_" + str(i)
    )(reshape)


    aspect_pool = layers.pooling.MaxPooling2D(
        pool_size = (training_inputs.shape[1] - filter_size + 1, 1),
        strides = (1,1),
        padding = 'valid',
        name = 'aspect_pool_' + str(i),
        data_format = "channels_last"
    )(aspect_conv)

    pooled_aspect_outputs.append(aspect_pool)

combined_pool = layers.Concatenate(axis=1)(pooled_aspect_outputs)
flat = layers.Flatten()(combined_pool)

h_drop = layers.Dropout(rate=dropout_keep_prob, name="dropout")(flat)

aspect_output = layers.Dense(train_label.shape[1], activation='sigmoid', name='main_output')(h_drop)

aspect_model = Model(inputs=inputs, outputs=aspect_output)

print('Compiling model...')
aspect_model.compile(
    optimizer = 'Adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
)

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "output", timestamp))
print("Saving to {}".format(out_dir))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_prefix = os.path.join(checkpoint_dir, "model")
checkpoints = callbacks.ModelCheckpoint(checkpoint_prefix + 'weights.{epoch:02d}.hdf5', monitor='val_loss,val_acc', verbose=0, save_best_only=False,mode='auto')
board = callbacks.TensorBoard(log_dir=out_dir)
print('Training aspect model...')
aspect_model.fit(training_inputs, train_label, epochs = num_epochs, batch_size = batch_size, callbacks=[checkpoints, board], validation_data=(evaluation_inputs, eval_label))

train_predictions = aspect_model.predict(training_inputs, batch_size = batch_size)

predicted_classes = []
true_treshold = 0.5

for prediction in train_predictions:
    class_no = 0
    predicted = []
    for prob in prediction:
        if prob > true_treshold:
            predicted.append(class_no)
        class_no += 1
    predicted_classes.append(predicted)


# Append aspects to text representations for sentiment polarity.
predicted_aspect_text_in = []
sentence_no = 0
for sentence in predicted_classes:
    # predicted_aspect_text_in.append(training_inputs[sentence_no][:])
    if len(sentence) == 0:
        predicted_aspect_text_in.append(np.append(training_inputs[sentence_no], 0))
    for label in sentence:
        predicted_aspect_text_in.append(np.append(training_inputs[sentence_no], label))
    # print(predicted_aspect_text_in[-1])
    sentence_no += 0

predicted_aspect_text_in = np.array(predicted_aspect_text_in)

# for i, filter_size in enumerate(sentiment_filter_dimensions):
#     filter_shape = [filter_size, embed_dim, 1, num_filters]
#
#     aspect_conv = layers.convolutional.Conv2D(
#         filters = num_filters,
#         kernel_size = (filter_size,embed_dim),
#         strides = (1,1),
#         padding = 'valid',
#         name = "sentiment_conv_" + str(i)
#     )(reshape)
#
#
#     aspect_pool = layers.pooling.MaxPooling2D(
#         pool_size = (predicted_aspect_text_in.shape[1] - filter_size + 1, 1),
#         strides = (1,1),
#         padding = 'valid',
#         name = 'sentiment_pool_' + str(i),
#         data_format = "channels_last"
#     )(aspect_conv)
#
#     pooled_aspect_outputs.append(aspect_pool)
#
# combined_pool = layers.Concatenate(axis=1)(pooled_aspect_outputs)
# flat = layers.Flatten()(combined_pool)
#
# h_drop = layers.Dropout(rate=dropout_keep_prob, name="dropout")(flat)
#
# aspect_output = layers.Dense(train_label.shape[1], activation='sigmoid', name='main_output')(h_drop)
#
# aspect_model = Model(inputs=inputs, outputs=aspect_output)

actual_classes = []
for sentence in train_label:
    # http://stackoverflow.com/a/6294205/3597964
    actual_classes.append([i for i, x in enumerate(sentence) if x == 1])

print(predicted_aspect_text_in.shape)
# print(predicted_classes)
# print(actual_classes)
# print(predicted_aspect_text_in)
