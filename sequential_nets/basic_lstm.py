"""
LSTM tutorial notes
source: https://becominghuman.ai/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow-1907a5bbb1fa
source: https://github.com/roatienza/Deep-Learning-Experiments/blob/master/Experiments/Tensorflow/RNN/rnn_words.py
"""

from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time 

start_time = time.time()

def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec/60) + " min"
    else:
        return str(sec/(60*60)) + " hr"


#Target log path
logs_path = '/tmp/tensorflow/rnn_words'
writer = tf.summary.FileWriter(logs_path)

#Text file containing words for training
training_file = 'belling_the_cat.txt'

#Read in data - convert into array of words
def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [word for i in range(len(content)) for word in content[i].split()]
    content = np.array(content)
    return content

training_data = read_data(training_file)
# print("Number of unique symbols: ", len(set(training_data))) => 112
print("Loaded training data ... ")

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dataset(training_data)
vocab_size = len(dictionary)

print("Dictionary: ", dictionary)
print('\n')
print("Reverse Dictionary: ", reverse_dictionary)


#Parameters
learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

#number of units in RNN cell
n_hidden = 512

#tf Graph input
x = tf.placeholder("float", [None, n_input, 1])
y = tf.placeholder("float", [None, vocab_size])

# RNN Output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}

def RNN(x, weights, biases):
    #reshape to [1, n_input]
    x = tf.reshape(x, [-1, n_input])

    # generate a n_input-element sequence of inputs
    x = tf.split(x, n_input, 1)

    #1-layer LSTM with n_hidden hidden units
    rnn_cell = rnn.BasicLSTMCell(n_hidden)

    #generate prediction
    outputs, states = rnn.static_rnn(rnn_cell,x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']



pred = RNN(x, weights, biases)

# Loss and optimizer


