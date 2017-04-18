# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import numpy as np
import itertools
import collections
import string
import unicodedata
import sys

# Fix absolute path
data_file = '/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_SB1_v2.xml'

def load_data(data_file):
    tree = ET.parse(data_file)
    root = tree.getroot()

    input_text = []
    output_labels = []
    meta = {'max_word_count': 0, 'max_string_length': 0}

    # Review = Review in root = Reviews
    for review in root:
        # sentence_data = []
        #
        for sentence in review.findall('sentences/sentence'):
            for text in sentence.findall('text'):
                input_text.append(text.text)

                if len(text.text) > meta['max_string_length']:
                    meta['max_string_length'] = len(text.text)
                if len(text.text.split(' ')) > meta['max_word_count']:
                    meta['max_word_count'] = len(text.text.split(' '))

            if len(sentence.findall('Opinions/Opinion')) == 0:
                output_labels.append([["NULL#NULL"]])
            else:
                labels = []
                for opinion in sentence.findall('Opinions/Opinion'):
                    op_atts = []

                    op_atts.append(opinion.attrib['category'])
                    op_atts.append(opinion.attrib['polarity'])

                    labels.append(op_atts)
                output_labels.append(labels)


    return [input_text, output_labels, meta]

def remove_outlying_labels(output_labels):

    # pruned_labels = output_labels

    label_list = {"OTHER#OTHER": 0}

    for element in output_labels:
        for quality in element:
            if quality[0] in label_list:
                label_list[quality[0]] += 1
            else:
                label_list[quality[0]] = 1



    # print label_list

    xx = 0

    while xx < len(output_labels):
        yy = 0
        while yy < len(output_labels[xx]):
            if label_list[output_labels[xx][yy][0]] < 20:
                output_labels[xx][yy][0] = "OTHER#OTHER"
            yy += 1
        xx += 1

    return output_labels


def boolean_labels(output_labels, return_index=False, label_list=None):
    """
    Format label data to be binary arrays.
    """

    # Populate label list if required, otherwise input is used (e.g. for
    # evaluationd data to follow same format as training)
    if label_list == None:
        label_list = ["OTHER#OTHER"]

        for element in output_labels:
            for quality in element:
                if quality[0] not in label_list and quality[0] != "NULL#NULL":
                    label_list.append(quality[0])

    labels_binary = []

    empty_label = []

    for element in label_list:
        empty_label.append(0)


    # TODO: Array of single aspect variable arrays.
    for sentence in output_labels:
        labels_binary.append(empty_label[:])
        for aspect in sentence:
            if aspect[0] in label_list:
                labels_binary[-1][label_list.index(aspect[0])] = 1
            elif aspect[0] == "NULL#NULL":
                labels_binary[-1] = empty_label[:]
            else:
                labels_binary[-1][label_list.index("OTHER#OTHER")] = 1
                # label_index[quality[0]] = label_index['max'] + 1
                # label_index['max'] += 1
                # labels_binary[-1][label_index[quality[0]]] = 1

    if return_index:
        # label list acts as a lookup incase of printing classification results
        return np.array(labels_binary), label_list
    else:
        return np.array(labels_binary)

def binary_sentiment(output_labels, return_index=False):

    sentiment_index = ['positive', 'conflict', 'negative']

    binary_sentiment = []

    empty_label = [0, 0, 0]

    for element in output_labels:
        binary_sentiment.append(empty_label[:])

        for example in element:
            if example[1] in sentiment_index:
                binary_sentiment[-1][sentiment_index.index(example[1])] = 1
            else:
                raise Exception('Mysterious 4th sentiment class')

    if return_index:
        return np.array(binary_sentiment), sentiment_index
    else:
        return np.array(binary_sentiment)

def boolean_combined(output_labels, return_index=False):

    binary_array = []

    # Setup sentiment index and empty array
    sentiment_index = ['positive', 'negative', 'other']

    boolean_labels = []

    empty_sentiment = [0, 0, 0]

    # Setup aspect index and empty array
    label_list = []

    for element in output_labels:
        for quality in element:
            if quality[0] not in label_list:
                label_list.append(quality[0])

    labels_binary = []

    empty_label = []

    for element in label_list:
        empty_label.append(0)

    combined_empty = [empty_label[:], empty_sentiment[:]]

    for review in output_labels:
        element = []

        for aspect in review:
            example = [empty_label[:], empty_sentiment[:]]

            # Probably if/except these
            example[0][label_list.index(aspect[0])] = 1
            if aspect[1] == 'neutral' or 'conflict':
                example[1][sentiment_index.index('other')] = 1
            else:
                example[1][sentiment_index.index(aspect[1])] = 1

            element.append(example)

        binary_array.append(element)


    # z = np.array(binary_array)

    # print z.shape
    return np.array(binary_array)

# def binary_eval(output_labels, label_list):


def return_batches(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1

    for epoch in range(num_epochs):

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


# http://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
translator = str.maketrans('','', string.punctuation)

def word_lists(text):
    output = []

    for sentence in text:
        sentence = sentence.replace(u'\xa0', u' ')
        output.append(sentence.translate(translator).split(' '))

    return output

def vocabulary_compile(text, max_length=None):

    words = word_lists(text)

    word_counts = collections.Counter(itertools.chain(*words))

    vocabulary = [x[0] for x in word_counts.most_common()]
    vocabulary = list(sorted(vocabulary))
    vocabulary.append('</NULL>')

    # vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    #
    # max_i = max(vocabulary[x] for x in vocabulary)
    # vocabulary['</NULL>'] =  max_i + 1
    return [vocabulary]

# FIXME TODO FIXME TODO FIXME TODO PADDING
def convert_text(sentences, vocabulary, meta, pad=True):
    training_data = []
    for sentence in sentences:
        sen_data = []
        sentence = sentence.replace(u'\xa0', u' ')
        sentence = sentence.translate(translator)
        for word in sentence.split(' '):
            if word in vocabulary:
                sen_data.append(vocabulary.index(word))
            else:
                sen_data.append(vocabulary.index('</NULL>'))
        yy = len(sentence.split(' '))
        while yy < meta['max_word_count']:
            sen_data.append(vocabulary.index('</NULL>'))
            yy += 1
        training_data.append(sen_data)

    # training_data = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])

    return np.array(training_data)

# x, y = load_data(data_file)
#
# alt_labels(y)

# x,y,z = load_data(data_file)

# print y

# print remove_outlying_labels(y)
#
# yi = boolean_labels(y)
#
# # print yi
# print len(x)
# print len(y)
# print len(yi)


#
# boolean_combined(y)

### FIXME
# if 0:
#     print(input_text)
#
#     print(output_labels)
#
#     print(count_label)
#     print(count_text)
#
#     print(max_length)
#
#
#     for key, value in sorted(label_count.iteritems(), key=lambda (k,v): (v,k)):
#         print "%s: %s" % (key, value)
#
#
#     for key, value in sorted(cat_count.iteritems(), key=lambda (k,v): (v,k)):
#         print "%s: %s" % (key, value)
#
#
#     for key, value in sorted(sub_type.iteritems(), key=lambda (k,v): (v,k)):
#         print "%s: %s" % (key, value)
