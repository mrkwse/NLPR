# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
import numpy as np

# Fix absolute path
data_file = '/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_English_SB2.xml'
tree = ET.parse('/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_English_SB2.xml')
root = tree.getroot()

input_text = []
output_labels = []

count_text = 0
count_label = 0
max_length = 0

label_count = {}
cat_count = {}
sub_type = {}


def load_data(data_file):
    tree = ET.parse(data_file)
    root = tree.getroot()

    input_text = []
    output_labels = []
    meta = {'max_word_count': 0, 'max_string_length': 0}

    for child in root:
        sentence_data = []
        for sentence in child.findall('sentences/sentence/text'):
            sentence_data.append(sentence.text)

            if len(sentence.text) > meta['max_string_length']:
                meta['max_string_length'] = len(sentence.text)
            if len(sentence.text.split(' ')) > meta['max_word_count']:
                meta['max_word_count'] = len(sentence.text.split(' '))

        input_text.append(sentence_data)

        label_data = []

        for opinion in child.findall('Opinions/Opinion'):
            opinion_el = []
            # opinion_el.append(opinion.attrib['category'].split('#')) # - To separate category and subtype
            opinion_el.append(opinion.attrib['category'])
            opinion_el.append(opinion.attrib['polarity'])

            label_data.append(opinion_el)

        output_labels.append(label_data)

    # TODO: Bin y = np.array(output_labels)

    # print y.shape

    return [input_text, output_labels, meta]


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
                if quality[0] not in label_list:
                    label_list.append(quality[0])

    labels_binary = []

    empty_label = []

    for element in label_list:
        empty_label.append(0)


    # TODO: Array of single aspect variable arrays.
    for element in output_labels:
        labels_binary.append(empty_label[:])
        for quality in element:
            if quality[0] in label_list:
                labels_binary[-1][label_list.index(quality[0])] = 1
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


    z = np.array(binary_array)

    # print z.shape
    return np.array(binary_array)

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


# x, y = load_data(data_file)
#
# alt_labels(y)

# x,y,z = load_data(data_file)
#
# boolean_combined(y)

### FIXME
if 0:
    print(input_text)

    print(output_labels)

    print(count_label)
    print(count_text)

    print(max_length)


    for key, value in sorted(label_count.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)


    for key, value in sorted(cat_count.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)


    for key, value in sorted(sub_type.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)
