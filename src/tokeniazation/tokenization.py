# Chapter 5, Natural Language Processing
"""
Example of TensorFlow Keras using preprocessing library
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?'
]

# A Tokenizer object that can tokenize 100 words
# this will be the maximum number of tokens to generate from the corpus of words
tokenizer = Tokenizer(num_words=100)

#create tokenized word index
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)


sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_sequences = tokenizer.texts_to_sequences(test_data)
word_index = tokenizer.word_index
print(word_index)
print(test_sequences)
## The output:
## [[1, 2, 3, 5], [7, 6]]
## So the new sentences, swapping back tokens for words, would be “today is a day”
# and “it rainy.”
## the context is lost, An out-of-vocabulary (OOV) might help here,
## and it can be set in the tokenizer
print('out-of-vocabulary example:')
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

test_sequences = tokenizer.texts_to_sequences(test_data)
word_index = tokenizer.word_index
print(word_index)
print(test_sequences)
##{'<OOV>': 1, 'today': 2, 'is': 3, 'a': 4, 'sunny': 5, 'day': 6, 'rainy': 7, 'it': 8}
##[[2, 3, 4, 1, 6], [1, 8, 1, 7, 1]]
##Your tokens list has a new item, “<OOV>,” and your test sentences maintain their length.
# ##Reverse-encoding them will now give “today is a <OOV> day” and
# “<OOV> it <OOV> rainy <OOV>