# hotel_reviews_sentiment_analysis

Implementation of Vanilla and Averaged Perceptron Algorithm for Hotel reviews classification as either truthful or deceptive, and either positive or negative.


## Dataset 

Dataset is adapted from the Deceptive Opinion Spam Corpus v1.

1. One file train-labeled.txt containing labeled training data with a single training instance (hotel review) per line (total 960 lines). The first 3 tokens in each line are: a unique 7-character alphanumeric identifier, label True or Fake and label Pos or Neg. These are followed by the text of the review.

2. One file dev-text.txt with unlabeled development data, containing just the unique identifier followed by the text of the review (total 320 lines).

3. One file dev-key.txt with the corresponding labels for the development data, to serve as an answer key.

## Evaluation

The final code trained model on the combined labeled training and development data, and tested the models on unseen data in a similar format.

Results (Vanilla model):
| Label  | Precision | Recall | F1 Score|
|--------|-----------|--------|---------|
|True | 0.86 | 0.91 | 0.88 | 
|Fake | 0.91 | 0.85 | 0.88 | 
|Pos | 0.93 | 0.89 | 0.91 | 
|Neg | 0.89 | 0.94 | 0.91 | 

Mean F1: 0.8968

Results (Averaged model):
| Label  | Precision | Recall | F1 Score|
|--------|-----------|--------|---------|
|True | 0.88 | 0.90 | 0.89 | 
|Fake | 0.90 | 0.88 | 0.89 | 
|Pos | 0.96 | 0.88 | 0.92 | 
|Neg | 0.89 | 0.96 | 0.92 | 

Mean F1: 0.9062

## Execution

Steps to execute

```python
# The learning program will be invoked in the following way:
python3 perceplearn.py /path/to/input

# The classification program will be invoked in the following way:
python3 percepclassify.py /path/to/model /path/to/input
```
