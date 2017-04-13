import xml.etree.ElementTree as ET

# Fix absolute path
tree = ET.parse('/Users/mrkwse/Documents/University/NLPR/OA/Data/ABSA16_Laptops_Train_English_SB2.xml')
root = tree.getroot()

input_text = []
output_labels = []

count_text = 0
count_label = 0
max_string_length = 0   # 4000
max_word_count = 0 # 73

label_count = {}
cat_count = {}
sub_type = {}

for child in root:
    sentence_data = []
    for sentence in child.findall('sentences/sentence/text'):
        sentence_data.append(sentence.text)
        count_text += 1

        if len(sentence.text) > max_string_length:
            max_string_length = len(sentence.text)

        if len(sentence.text.split(' ')) > max_word_count:
            max_word_count = len(sentence.text.split(' '))


    input_text.append(sentence_data)

    label_data = []

    for opinion in child.findall('Opinions/Opinion'):
        opinion_el = []
        opinion_el.append(opinion.attrib['category'].split('#'))
        opinion_el.append(opinion.attrib['polarity'])

        if opinion.attrib['category'] in label_count:
            label_count[opinion.attrib['category']] += 1
        else:
            label_count[opinion.attrib['category']] = 1

        if opinion.attrib['category'].split('#')[0] in cat_count:
            cat_count[opinion.attrib['category'].split('#')[0]] += 1
        else:
            cat_count[opinion.attrib['category'].split('#')[0]] = 1

        if opinion.attrib['category'].split('#')[1] in sub_type:
            sub_type[opinion.attrib['category'].split('#')[1]] += 1
        else:
            sub_type[opinion.attrib['category'].split('#')[1]] = 1

        count_label += 1

        label_data.append(opinion_el)

    output_labels.append(label_data)


print(input_text)

print(output_labels)

print(count_label)
print(count_text)

print(max_string_length)

print(max_word_count)

if 0:
    for key, value in sorted(label_count.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)


    for key, value in sorted(cat_count.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)


    for key, value in sorted(sub_type.iteritems(), key=lambda (k,v): (v,k)):
        print "%s: %s" % (key, value)
