import keras.layers as layers
from keras.models import Model
import keras.preprocessing.text as text
import os
import numpy as np
import preprocessing

vocab_size = 100
embed_dim = 50
num_filters = 256
aspect_filter_dimensions = [2,3,4]

batch_size = 64
num_epochs = 200

training_data_path = os.environ['TRAIN_PATH'] #/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_SB1_v2.xml
evaluation_data_path = os.environ['EVAL_PATH'] #/Users/mrkwse/Documents/University/NLPR/OA/Data/EN_LAPT_SB1_TEST_.xml.gold

print('Loading data...')
text_train, label_train, train_meta = preprocessing.load_data(training_data_path)
text_eval, label_eval, eval_meta = preprocessing.load_data(evaluation_data_path)

combined_text = text_train + text_eval

print('Building vocabulary...')
vocabulary, vocabulary_inv = preprocessing.vocabulary_transform(combined_text)
print('Vocabulary size' + str(len(vocabulary)))

train_label, label_index = preprocessing.binary_labels(label_train, return_index=True)
eval_label = preprocessing.binary_labels(label_eval, label_list = label_index)

tokenizer = text.Tokenizer(num_words=train_meta['max_word_count'])
# tokenizer = text.Tokenizer()

tokenizer.fit_on_texts(text_train)

# train_in = tokenizer.texts_to_sequences(text_train)
# eval_in = tokenizer.texts_to_sequences(text_eval)

training_inputs = preprocessing.build_input_data(text_train, vocabulary, train_meta)
evaluation_inputs = preprocessing.build_input_data(text_eval, vocabulary, train_meta)

print('Preprocessing complete.')


# This shouldn't be hard coded
inputs = layers.Input(shape=(training_inputs.shape))

embedded_chars = layers.Embedding(
                    input_dim = train_meta['max_word_count'],
                    output_dim = embed_dim,
                    input_length = train_meta['max_string_length']
                 )(inputs)

# TODO[deletecomment] Possibly for loop through multiple filter shapes

pooled_aspect_outputs = []

for i, filter_size in enumerate(aspect_filter_dimensions):
    # filter_shape = [filter_size, embed_dim, 1, num_filters]

    aspect_conv = layers.convolutional.Conv2D(
        filters = filter_size,
        kernel_size = (1,1),
        strides = (1,1),
        padding = 'valid'
        # name = "aspect_conv"
    )(embedded_chars)
    # aspect_conv = layers.convolutional.Conv2D(
    #     filters = num_filters,
    #     kernel_size = [1,1],
    #     strides = (1,1),
    #     padding = 'valid',
    #     name = "aspect_convolution"
    # )(embedded_chars)

    aspect_pool = layers.pooling.MaxPooling2D(
        pool_size = train_meta['max_word_count'] - filter_size + 1,
        strides = (1,1),
        padding = 'valid',
        name = 'aspect_pool',
        data_format = 'channels_last'
    )(aspect_conv)

    pooled_aspect_outputs.append(aspect_conv)

total_filters = num_filters * len(aspect_filter_dimensions)
# h_pool = layers.concatenate(pooled_aspect_outputs, 3)
h_pool = layers.Concatenate(axis=2)(pooled_aspect_outputs)
main_output = layers.Dense(1, activation='sigmoid', name='main_output')(h_pool)


model = Model(inputs=inputs, outputs=h_pool)


model.compile(
    optimizer = 'Adam',
    loss = 'binary_crossentropy',
    metrics = ['accuracy']
    #loss_weights = [a,b]
)


model.fit(training_inputs, train_label, epochs = num_epochs, batch_size = batch_size)

# def train_step(text_batch, label_batch):
#     """
#     Single training step
#     """


# batches = preprocessing.return_batches(
#     zip(train_in, train_label),
#     batch_size,
#     num_epochs
# )
