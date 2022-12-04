import numpy as np
import re
import json
import sys

stop_words = {"a",  "about",  "above",  "after",  
              "again",  "ain",  "all",  "am",  
              "an",  "and",  "any",  "aren",  
              "as",  "at",  "be",  "because",  
              "been",  "before",  "being",  "below",  
              "between",  "both",  "but",  "by",  
              "couldn",  "didn",  "doesn",  "don",  
              "if",  "in",  "into",  "is",  "isn",   "for",  "from",
              "isn't",  "it",  "it's",  "its",  "itself",  
              "just",  "me",  "my",  "myself",  "no",  
              "nor",  "not",  "now",  "of",  "off",  
              "on",  "once",  "only",  "or",  "other",  
              "our",  "ours",  "ourselves",  "out",  "over",  
              "own",  "same",  "shan",  "shan't",  "down",  "each",
              "she",  "she's",  "so",  "some",  "such",  "than",  
              "that",  "that'll",  "the",  "their",  "theirs",  
              "them",  "themselves",  "then",  "there",  "these",  
              "they",  "this",  "those",  "through",  "to",  
              "too",  "under",  "until",  "up",  "we",  "what",  
              "when",  "where",  "which",  "while",  "who",  
              "whom",  "why",  "will",  "with",  "won",  "you",  
              "you'd",  "you'll",  "you're",  "you've",  "your",  
              "yours",  "yourself",  "yourselves",   
              "hadn",  "hasn",  "haven",  "having",  
              "he",  "her",  "here",  "hers",  "herself",  
              "him",  "himself",  "his",  "how",  "i"  }

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

def vanilla_averaged_perceptron(features, maxIter):
    weights = np.zeros([features.shape[1] - 1, 1], dtype = np.float64)
    bias = 0
    weights_avg = np.zeros([features.shape[1] - 1, 1], dtype = np.float64)
    bias_avg = 0
    counter = 1
    iter_val = 1
    while iter_val <= maxIter:
        for data_point in features:
            x = data_point[:len(data_point)-1]
            y = data_point[len(data_point)-1]
            x = x.reshape(x.shape[0], -1)
            a = np.dot(weights.transpose(), x) + bias
            if (y*a <= 0):
                weights = weights + y*x
                bias = bias + y
                weights_avg = weights_avg + y*counter*x
                bias_avg = bias_avg + y*counter
            counter = counter + 1
        iter_val = iter_val + 1

    return weights, bias, (weights - ((1/counter)*weights_avg)), (bias - ((1/counter)*bias_avg))

def create_features(sentences, vocab_position):
    arr_fake_true = np.zeros([0,len(vocab_position)+1], dtype = np.float64)
    arr_neg_pos = np.zeros([0,len(vocab_position)+1], dtype = np.float64)

    for sentence in sentences:
        sentence = sentence.strip()
        review_data = sentence.split(" ", 3)
        review = review_data[3]
        review = handle_contractions(review)
        review = re.sub(r'[^\w\s]'," ", review)
        review = re.sub("\s\s+", " ", review)
        review_tokens = review.split()
        review_feature = np.zeros(len(vocab_position)+1)
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
        fake_true_label = review_data[1]
        neg_pos_label = review_data[2]
        label = 1
        if fake_true_label == "Fake":
            label = -1
        review_feature[len(review_feature) - 1] = label
        arr_fake_true = np.append(arr_fake_true, review_feature.reshape(-1,review_feature.shape[0]), axis=0)
        label = 1
        if neg_pos_label == "Neg":
            label = -1
        review_feature[len(review_feature) - 1] = label
        arr_neg_pos = np.append(arr_neg_pos, review_feature.reshape(-1,review_feature.shape[0]), axis=0)
    return arr_fake_true, arr_neg_pos

def create_vocabulary(sentences):
    vocab_position = {}
    vocab_dict = {}
    for sentence in sentences:
        sentence = sentence.strip()
        review = sentence.split(" ", 3)[3]
        review = handle_contractions(review)
        review = re.sub(r'[^\w\s]', " ", review)
        review = re.sub("\s\s+", " ", review)
        review_tokens = review.split()
        review_tokens = list(set(review_tokens).difference(stop_words))
        
        for token in review_tokens:
            vocab_dict[token] = vocab_dict.get(token, 0) + 1
    
    key_to_delete = max(vocab_dict, key=lambda k: vocab_dict[k])
    del vocab_dict[key_to_delete]
    vocab = sorted(vocab_dict, key=vocab_dict.get, reverse=True)
    
    vocab_position = {k: v for v, k in enumerate(vocab)}
    return vocab_dict, vocab_position

def read_file(file_path):
    sentences = []
    with open(file_path) as fp:
        sentences = fp.readlines()
    return sentences

file_path = sys.argv[1]
sentences = read_file(file_path)
vocab_dict, vocab_position = create_vocabulary(sentences)
arr_fake_true, arr_neg_pos = create_features(sentences, vocab_position)
weights_np_vanilla, bias_np_vanilla, weights_np_averaged, bias_np_averaged = vanilla_averaged_perceptron(arr_neg_pos, 10)
weights_ft_vanilla, bias_ft_vanilla, weights_ft_averaged, bias_ft_averaged = vanilla_averaged_perceptron(arr_fake_true, 10)


parameters = []
parameters.append("weights_np" + " " + str(weights_np_vanilla.tolist()))
parameters.append("bias_np" + " " + str(bias_np_vanilla))
parameters.append("weights_ft" + " " + str(weights_ft_vanilla.tolist()))
parameters.append("bias_ft" + " " + str(bias_ft_vanilla))
parameters.append("vocab_position" + " " + json.dumps(vocab_position))

with open("vanillamodel.txt", "w") as outfile:
    outfile.write("\n".join(parameters))


parameters = []
parameters.append("weights_np" + " " + str(weights_np_averaged.tolist()))
parameters.append("bias_np" + " " + str(bias_np_averaged))
parameters.append("weights_ft" + " " + str(weights_ft_averaged.tolist()))
parameters.append("bias_ft" + " " + str(bias_ft_averaged))
parameters.append("vocab_position" + " " + json.dumps(vocab_position))

with open("averagedmodel.txt", "w") as outfile:
    outfile.write("\n".join(parameters))
