import numpy as np

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


def add_class_to_text(predicted_classes, text_input, aspect_counts):

    # Append aspects to text representations for sentiment polarity.
    predicted_aspect_text_in = []
    sentence_no = 0
    while sentence_no < len(predicted_classes):
        aspect = 0
        while aspect < aspect_counts[sentence_no]:
            if len(predicted_classes[sentence_no]) == 0:
                predicted_aspect_text_in.append(np.append(text_input[sentence_no], 0))
            else:
                if len(predicted_classes[sentence_no]) < aspect_counts[sentence_no]:
                    predicted_aspect_text_in.append(np.append(text_input[sentence_no], 0))
                else:
                    predicted_aspect_text_in.append(np.append(text_input[sentence_no], predicted_classes[sentence_no][aspect]))
            aspect += 1
        sentence_no += 1

    return np.array(predicted_aspect_text_in)
