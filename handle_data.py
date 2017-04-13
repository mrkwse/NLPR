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

    for child in root:
        sentence_data = []
        for sentence in child.findall('sentences/sentence/text'):
            sentence_data.append(sentence.text)

        input_text.append(sentence_data)

        label_data = []

        for opinion in child.findall('Opinions/Opinion'):
            opinion_el = []
            # opinion_el.append(opinion.attrib['category'].split('#')) # - To separate category and subtype
            opinion_el.append(opinion.attrib['category'])
            opinion_el.append(opinion.attrib['polarity'])

            label_data.append(opinion_el)

        output_labels.append(label_data)

    return [input_text, output_labels]


def binary_labels(output_labels, return_index=False):
    """
    Format label data to be binary arrays.
    """

    label_list = []

    for element in output_labels:
        for quality in element:
            if quality[0] not in label_list:
                label_list.append(quality[0])

    labels_binary = []

    empty_label = []

    for element in label_list:
        empty_label.append(0)

    for element in output_labels:
        labels_binary.append(empty_label[:])
        for quality in element:
            if quality[0] in label_list:
                labels_binary[-1][label_list.index(quality[0])] = 1
            else:
                raise Exception('Missing label in list')
                # label_index[quality[0]] = label_index['max'] + 1
                # label_index['max'] += 1
                # labels_binary[-1][label_index[quality[0]]] = 1

    if return_index:
        # label list acts as a lookup incase of printing classification results
        return labels_binary, label_list
    else:
        return labels_binary

def binary_sentiment(output_labels, return_index=false):

    sentiment_index = ['positive', 'negative', 'conflict']

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
        return binary_sentiment, sentiment_index
    else:
        return binary_sentiment

# x, y = load_data(data_file)
#
# alt_labels(y)




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
