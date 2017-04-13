import xml.etree.ElementTree as ET

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
            opinion_el.append(opinion.attrib['category'].split('#'))
            opinion_el.append(opinion.attrib['polarity'])

            label_data.append(opinion_el)

        output_labels.append(label_data)


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
