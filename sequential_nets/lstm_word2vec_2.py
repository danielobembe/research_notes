"""
LSTM tutorial  - using LSTM with pre-trained word2vec.
source: http://www.brightideasinanalytics.com/pretrained-word-vectors-example/
source: http://www.brightideasinanalytics.com/rnn-pretrained-word-vectors/
glove source: https://www.kaggle.com/rtatman/glove-global-vectors-for-word-representation#glove.6B.50d.txt
"""

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
file.close()

# print("man: ", embedding_dict['man'])

from scipy import spatial
import numpy as np 

king_vector = np.array(embedding_dict['king'])
man_vector  = np.array(embedding_dict['man'])
woman_vector = np.array(embedding_dict['woman'])

new_vector = king_vector - man_vector + woman_vector

tree = spatial.KDTree(glove_embed)

nearest_dist, nearest_idx = tree.query(new_vector, 10)
nearest_words = [glove_vocab[i] for i in nearest_idx]
print("Nearest words to 'king - man + woman: ", nearest_words)