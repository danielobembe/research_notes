"""
LSTM tutorial  - using LSTM with pre-trained word2vec.
source: http://www.brightideasinanalytics.com/pretrained-word-vectors-example/
source: http://www.brightideasinanalytics.com/rnn-pretrained-word-vectors/
glove_source: https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation#glove.6B.50d.txt
"""
import tensorflow as tf
from tensorflow.contrib import rnn 
import numpy as np 
import collections
import random 
from scipy import spatial

#Load glove vectors
filepath_glove = 'glove.6B.50d.txt'
glove_vocab     = []
glove_embed     = []
embedding_dict  = {}

file = open(filepath_glove, 'r', encoding='UTF-8')
for line in file.readlines():
    row = line.strip().split(' ')
    vocab_word = row[0]
    glove_vocab.append(vocab_word)
    embed_vector = [float(i) for i in row[1:]]
    embedding_dict[vocab_word] = embed_vector
file.close()
print('Loaded GloVe')

glove_vocab_size    = len(glove_vocab)
embedding_dim       = len(embed_vector)


#Load our text/corpus
fable_text = ""
with open("belling_the_cat.txt", 'r') as myfile:
    fable_text = myfile.read()

def read_data(raw_text):
    content = raw_text 
    content = content.split()
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content 

training_data = read_data(fable_text)
print(training_data)

def build_dictionaries(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count: 
        dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dictionaries(training_data

"""
Important objects we now have:
embedding_dict = dictionary containing "word":"embedding" pairs, encountered during pre-training
dictionary = words from current training documents : unique integers
reverse_dictionary = unique integer : words from current training docs 
"""

#Create our embedding array (for our corpus, using glove vectors):
doc_vocab_size = len(dictionary)
dict_as_list = sorted(dictionary.items(), key = lambda x : x[1])
embeddings_tmp = []

for i in range(doc_vocab_size):
    item = dict_as_list[i][0]
    #if word is already in glove, store it's vector representation
    if item in glove_vocab:
        embeddings_tmp.append(embedding_dict[item])
    #else generate a random vector as it's vector representation
    else:
        rand_num = np.random.uniform(low=-0.2, hig0.2, size=embedding_dim)
        embeddings_tmp.append(rand_num)
embedding = np.asarray(embeddings_tmp)
#create tree to later search for closest vectors to predictions
tree = spatial.KDTree(embedding)


#Set up RNN model




