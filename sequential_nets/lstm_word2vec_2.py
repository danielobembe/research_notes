"""
LSTM tutorial  - using LSTM with pre-trained word2vec.
source: http://www.brightideasinanalytics.com/pretrained-word-vectors-example/
source: http://www.brightideasinanalytics.com/rnn-pretrained-word-vectors/
glove source: https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation#glove.6B.50d.txt
"""
from __future__ import print_function
import tensorflow as tf 
from tensorflow.contrib import rnn
import random
from scipy import spatial
import numpy as np 
import collections


#Note: word embeddings === word vectors
filename = 'glove.6B.50d.txt'


glove_vocab     = []    #list of words we have embeddings for
glove_embed     = []    #list of lists containing embedding vectors 
embedding_dict  = {}    #dict{ word : embedding }

file = open(filename, 'r', encoding='UTF-8')

for line in file.readlines():
    row = line.strip().split(' ')
    vocab_word = row[0]
    glove_vocab.append(vocab_word)
    embed_vector = [float(i) for i in row[1:]]
    embedding_dict[vocab_word] = embed_vector
    glove_embed.append(embed_vector)

print('Loaded GLOVE\n')
embedding_dim = len(embed_vector)
file.close()

# print("man: ", embedding_dict['man'])
fable_text = """
long ago , the mice had a general council to consider what measures
they could take to outwit their common enemy , the cat . some said
this , and some said that but at last a young mouse got up and said
he had a proposal to make , which he thought would meet the case . 
you will all agree , said he , that our chief danger consists in the
sly and treacherous manner in which the enemy approaches us . now , 
if we could receive some signal of her approach , we could easily
escape from her . i venture , therefore , to propose that a small
bell be procured , and attached by a ribbon round the neck of the cat
. by this means we should always know when she was about , and could
easily retire while she was in the neighbourhood . this proposal met
with general applause , until an old mouse got up and said that is
all very well , but who is to bell the cat ? the mice looked at one
another and nobody spoke . then the old mouse said it is easy to
propose impossible remedies .
"""

fable_text = fable_text.replace('\n','')

def read_data(raw_text):
    content = raw_text
    content = content.split() #splits the text by spaces (default split character)
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content
training_data = read_data(fable_text)

def build_dictionaries(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_dictionaries(training_data)

doc_vocab_size = len(dictionary)
dict_as_list = sorted(dictionary.items(), key = lambda x : x[1] )
embeddings_tmp = []

for i in range(doc_vocab_size):
    item = dict_as_list[i][0]
    if item in glove_vocab:
        embeddings_tmp.append(embedding_dict[item])
    else:
        rand_num = np.random.uniform(low=-0.2, high=0.2,size=embedding_dim)
        embeddings_tmp.append(rand_num)
embedding = np.asarray(embeddings_tmp)
tree = spatial.KDTree(embedding)


#model paramters
learning_rate   = 0.001
n_input         = 3         #number of words read in at a time
n_hidden        = 512

#create input placeholders
x = tf.placeholder(tf.int32, [None, n_input])
y = tf.placeholder(tf.float32, [None, embedding_dim])

#RNN output node weights and biases
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, embedding_dim]))
}
biases = {
    'out': tf.Variable(tf.random_normal([embedding_dim]))
}

with tf.name_scope("embedding"):
    W = tf.Variable(tf.constant(0.0, shape=[doc_vocab_size, embedding_dim]),
                    trainable=True, name='W')
    embedding_placeholder = tf.placeholder(tf.float32, [doc_vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    embedded_chars = tf.nn.embedding_lookup(W,x)

#reshape input data
x_unstack = tf.unstack(embedded_chars, n_input, 1)

#create RNN cells
rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(n_hidden), rnn.BasicLSTMCell(n_hidden)])
outputs, states = rnn.static_rnn(rnn_cell, x_unstack, dtype=tf.float32)

#capture last output
pred = tf.matmul(outputs[-1], weights['out']) + biases['out']

# create loss function and optimizer
cost = tf.reduce_mean(tf.nn.l2_loss(pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

step = 0
offset = random.randint(0, n_input+1)
end_offset = n_input + 1
acc_total = 0
loss_total = 0
training_iters = 100
display_step = 500

while step < training_iters:
    #generate minibatch
    if offset > (len(training_data) - end_offset):
        offset = random.randint(0, n_input+1)

    # create integer representations for input words
    x_integers = [[ dictionary[str(training_data[i])]] for i in range(offset, offset+n_input)]
    x_integers = np.reshape(np.array(x_integers), [-1, n_input])

    # create embedding for target vector
    y_position = offset + n_input 
    y_integer = dictionary[training_data[y_position]]
    y_embedding = embedding[y_integer, :]
    y_embedding = np.reshape(y_embedding,[1,-1])

    _, loss, pred_ = sess.run([optimizer, cost, pred], feed_dict = {
        x: x_integers,
        y: y_embedding
    })
    loss_total += loss

    ## Display output
    words_in = [str(training_data[i]) for i in range(offset, offset+n_input)]
    target_word = str(training_data[y_position])
    nearest_dist, nearest_idx = tree.query(pred_[0], 3)
    nearest_words = [reverse_dictionary[idx] for idx in nearest_idx]
    print("%s - [%s] vs [%s]" % (words_in, target_word, nearest_words))
    print("Average Loss= " + "{:.6f}".format(loss_total/display_step))
    offset += (n_input + 1)
    ##

    
    #me
    step += 1
    print(loss)
print("Finished Optimization")   

    
