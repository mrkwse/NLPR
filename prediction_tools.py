
def predict_classes(predicted_values, true_treshold=0.5):

    predicted_classes = []

    for prediction in predicted_values:
        class_no = 0
        predicted = []
        for prob in prediction:
            if prob > true_treshold:
                predicted.append(class_no)
            class_no += 1
        predicted_classes.append(predicted)

    return predicted_classes


def add_class_to_text(predicted_classes, text_input):

    # Append aspects to text representations for sentiment polarity.
    predicted_aspect_text_in = []
    sentence_no = 0
    for sentence in predicted_classes:
        # predicted_aspect_text_in.append(training_inputs[sentence_no][:])
        if len(sentence) == 0:
            predicted_aspect_text_in.append(np.append(text_input[sentence_no], 0))
        for label in sentence:
            predicted_aspect_text_in.append(np.append(text_input[sentence_no], label))
        # print(predicted_aspect_text_in[-1])
        sentence_no += 0

        return predicted_aspect_text_in
