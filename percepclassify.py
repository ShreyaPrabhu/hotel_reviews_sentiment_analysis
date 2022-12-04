import ast
import sys
import numpy as np
import re

parameters_dict = {}

def handle_contractions(review):
    review = re.sub(r"won\'t", "will not", review)
    review = re.sub(r"can\'t", "can not", review)
    review = re.sub(r"n\'t", " not", review)
    review = re.sub(r"\'t", " not", review)
    review = re.sub(r"\'d", " would", review)
    review = re.sub(r"\'ll", " will", review)
    review = re.sub(r"\'ve", " have", review)
    review = re.sub(r"\'m", " am", review)
    review = re.sub(r"\'re", " are", review)
    review = re.sub(r"\'s", " is", review)
    return review

def create_features_to_predict(sentences, vocab_position):
    test_identifiers = []
    test_features = np.zeros([0,len(vocab_position)], dtype = np.float64)
    for sentence in sentences:
        sentence = sentence.strip()
        review_data = sentence.split(" ", 1)
        test_identifiers.append(review_data[0])
        review = review_data[1]
        review = handle_contractions(review)
        review = re.sub(r'[^\w\s]', " ", review)
        review = re.sub("\s\s+", " ", review)
        review_tokens = review.split()
        review_feature = np.zeros(len(vocab_position))
        review_token_counts = {}
        for token in review_tokens:
            if token in review_token_counts:
                count = review_token_counts[token]
                review_token_counts[token] = count + 1
            else:
                review_token_counts[token] = 1
        
        for token in review_tokens:
            index = vocab_position.get(token, -1)
            if index == -1:
                continue 
            else:
                np.put(review_feature, [index], review_token_counts[token])

        test_features = np.append(test_features, review_feature.reshape(-1,review_feature.shape[0]), axis=0)
    return test_features, test_identifiers

def predict(file_path):
    sentences = []
    with open(file_path) as fp:
        sentences = fp.readlines()
    test_features, test_identifiers = create_features_to_predict(sentences, parameters_dict["vocab_position"])
    prediction_np = np.dot(test_features, np.array(parameters_dict["weights_np"])) + parameters_dict["bias_np"]
    prediction_ft = np.dot(test_features, np.array(parameters_dict["weights_ft"])) + parameters_dict["bias_ft"]
    return prediction_np, prediction_ft, test_identifiers

file_path_model = sys.argv[1]
file_path = sys.argv[2]
with open(file_path_model) as fp:
    sentences = fp.readlines()
    for sentence in sentences:
        data = sentence.split(" ", 1)
        parameters_dict[data[0]] = ast.literal_eval(data[1])

prediction_np, prediction_ft, test_identifiers = predict(file_path)
prediction_np_list = prediction_np.tolist()
prediction_np_list = [item for sublist in prediction_np_list for item in sublist]
prediction_np_class = ["Pos" if val>=0 else "Neg" for val in prediction_np_list]

prediction_ft_list = prediction_ft.tolist()
prediction_ft_list = [item for sublist in prediction_ft_list for item in sublist]
prediction_ft_class = ["True" if val>=0 else "Fake" for val in prediction_ft_list]
result_to_save = [idf + " " + np + " " + ft for idf, np, ft in zip(test_identifiers, prediction_ft_class, prediction_np_class)]
with open("percepoutput.txt", "w") as outfile:
    outfile.write("\n".join(result_to_save))
