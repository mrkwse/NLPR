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


def binary_labels(output_labels, return_index=False, label_list=None):
    """
    Format label data to be binary arrays.
    """

    # Populate label list if required, otherwise input is used (e.g. for
    # evaluationd data to follow same format as training)
    if label_list == None:
        label_list = ["", "OTHER#OTHER"]

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

    sentiment_index = ['positive', 'neutral', 'negative']

    binary_sentiment = []

    empty_label = [0, 0, 0]

    for element in output_labels:
        element = []

        for example in element:
            if example != ['NULL#NULL']:
                label = empty_label[:]
                if example[1] in sentiment_index:
                    label[sentiment_index.index(example[1])] = 1
                else:
                    raise Exception('Mysterious 4th sentiment class')
                element.append(label)
        binary_sentiment.append(element)

    if return_index:
        return np.array(binary_sentiment), sentiment_index
    else:
        return np.array(binary_sentiment)

def binary_combined(output_labels, return_index=False):

    binary_array = []

    # Setup sentiment index and empty array
    sentiment_index = ['positive', 'negative', 'other']

    binary_labels = []

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


# http://stackoverflow.com/questions/34293875/how-to-remove-punctuation-marks-from-a-string-in-python-3-x-using-translate
translator = str.maketrans('','', string.punctuation)

def word_lists(text):
    output = []

    for sentence in text:
        sentence = sentence.replace(u'\xa0', u' ')
        output.append(sentence.translate(translator).split(' '))

    return output


# TODO Expand FOR loops
def vocabulary_transform(text, max_length=None):

    words = word_lists(text)

    word_counts = collections.Counter(itertools.chain(*words))

    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # for word in word_counts.most_common():
    #     vocabulary_inv.append(word[0])
    vocabulary_inv = list(sorted(vocabulary_inv))
    vocabulary_inv.append('</NULL>')

    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    max_i = max(vocabulary[x] for x in vocabulary)
    vocabulary['</NULL>'] =  max_i + 1
    return [vocabulary, vocabulary_inv]

# FIXME TODO FIXME TODO FIXME TODO PADDING
def build_input_data(sentences, vocabulary, meta, pad=True):
    training_data = []
    for sentence in sentences:
        sen_data = []
        sentence = sentence.replace(u'\xa0', u' ')
        sentence = sentence.translate(translator)
        for word in sentence.split(' '):
            if word in vocabulary:
                sen_data.append(vocabulary[word])
            else:
                sen_data.append(vocabulary['</NULL>'])
        yy = len(sentence.split(' '))
        while yy < meta['max_word_count']:
            sen_data.append(vocabulary['</NULL>'])
            yy += 1
        training_data.append(sen_data)

    return np.array(training_data)

def sort_array(array):
    sorted_array = []

    while len(array) > 0:
        min

def binary_combined(labels_in, aspect_index):
    output = []

    empty_aspect = []

    for element in aspect_index:
        empty_aspect.append(0)

    # Skip first element for empty (when converting to int)
    sentiment_index = {'positive': 1, 'neutral': 2, 'negative': 3}

    empty_sentiment = [0, 0, 0, 0]

    for review in labels_in:
        review_representation = []
        for aspect in review:
            aspect_representation = [empty_aspect[:], empty_sentiment[:]]

            if aspect[0] in aspect_index:
                aspect_representation[0][aspect_index.index(aspect[0])] = 1
            elif aspect[0] == "NULL#NULL":
                aspect_representation[0] = empty_aspect[:]
            else:
                aspect_representation[0][aspect_index.index("OTHER#OTHER")]

            if len(aspect) == 2:
                if aspect[1] in sentiment_index:
                    aspect_representation[1][sentiment_index[aspect[1]]] = 1

            review_representation.append(aspect_representation)

        output.append(review_representation)

    return(np.array(output))


def sort_array(array):
    sorted_array = []

    while len(array) > 0:
        min_class = min([x[0] for x in array])

        e_x = 0
        for element in array:
            if element[0] == min_class:
                sorted_array.append(element)
                array.pop(e_x)
            e_x +=1


    return sorted_array

def binary_to_int(binary_labels_in):
    binary_sentiment = []
    binary_sentiment_expanded = []
    int_pairs = []
    int_pairs_expanded = []

    for review in binary_labels_in:
        binary_review = []
        int_review = []

        for aspect in review:
            aspect_binary_arr = []
            aspect_int_arr = []

            if 1 in aspect[0]:
                aspect_int = aspect[0].index(1)
                aspect_binary_arr.append(aspect_int)
                aspect_int_arr.append(aspect_int)
            else:
                aspect_binary_arr.append(0)
                aspect_int_arr.append(0)

            aspect_binary_arr.append(aspect[1])

            if 1 in aspect[1]:
                aspect_int_arr.append(aspect[1].index(1))
            else:
                aspect_int_arr.append(0)

            # binary_sentiment_expanded.append(aspect_binary_arr)
            # int_pairs_expanded.append(aspect_int_arr)

            binary_review.append(aspect_binary_arr)
            int_review.append(aspect_int_arr)

        binary_review = sort_array(binary_review)
        int_review = sort_array(int_review)


        binary_sentiment_expanded += binary_review

        int_pairs_expanded += int_review

        binary_sentiment.append(binary_review)
        int_pairs.append(int_review)
        # binary_sentiment = np.append(binary_sentiment, binary_review, axis=0)
        # int_pairs = np.append(int_pairs, int_review, axis=0)

    return binary_sentiment, binary_sentiment_expanded, int_pairs, int_pairs_expanded


# def isolate_binary_sentiment(sorted_labels_in):
