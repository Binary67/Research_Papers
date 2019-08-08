import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Read the corpus from txt file
READFILE = open('corpus.txt', 'r')
CORPUS = READFILE.read()
CORPUS = CORPUS.lower()

WORDS = []
SENTENCES = []
NEW_CORPUS = []


STOP_WORDS = set(stopwords.words('english'))

# Tokenize the words and remove stopwords
for SENTENCE in CORPUS.split(','):
    TEMP = SENTENCE.split()
    TEMP_1 = []
    for i in TEMP:
        if i not in STOP_WORDS:
            TEMP_1.append(i)
    SENTENCES.append(TEMP_1)
    for WORD in SENTENCE.split():
        if WORD not in STOP_WORDS:
            WORDS.append(WORD)

# Remove all duplicated words
WORDS = set(WORDS)

VOCAB_SIZE = len(WORDS)

# Create two dictionaries
WORD2INT = {}
INT2WORD = {}

for i, WORD in enumerate(WORDS):
    WORD2INT[WORD] = i
    INT2WORD[i] = WORD

# Create window size
WINDOW_SIZE = 2
DATA = []

# Slide the window
for SENTENCE in SENTENCES:
    for WORD_INDEX, WORD in enumerate(SENTENCE):
        for NEIGHBOUR_WORD in SENTENCE[max(WORD_INDEX - WINDOW_SIZE, 0) : min(WORD_INDEX + WINDOW_SIZE, len(SENTENCE)) + 1]:
            if NEIGHBOUR_WORD != WORD:
                DATA.append([WORD, NEIGHBOUR_WORD])

# One hot encoding
def One_Hot_Encoding(DATA_INDEX, VOCAB_SIZE):
    TEMP = np.zeros(VOCAB_SIZE)
    TEMP[DATA_INDEX] = 1
    return TEMP

X_TRAIN = []
Y_TRAIN = []

for WORD_DATA in DATA:
    X_TRAIN.append(One_Hot_Encoding(WORD2INT[WORD_DATA[0]], VOCAB_SIZE))
    Y_TRAIN.append(One_Hot_Encoding(WORD2INT[WORD_DATA[1]], VOCAB_SIZE))

# Convert to numpy array
X_TRAIN = np.asarray(X_TRAIN)
Y_TRAIN = np.asarray(Y_TRAIN)

# Tensorflow implementation
X_INPUT = tf.placeholder(tf.float32, shape = (None, VOCAB_SIZE))
Y_INPUT = tf.placeholder(tf.float32, shape = (None, VOCAB_SIZE))

EMBEDDING_SIZE = 5

# Layer 1
WEIGHT_1 = tf.Variable(tf.random_normal([VOCAB_SIZE, EMBEDDING_SIZE]))
BIASE_1 = tf.Variable(tf.random_normal([EMBEDDING_SIZE]))
Z1 = tf.add(tf.matmul(X_INPUT, WEIGHT_1), BIASE_1)

# Layer 2
WEIGHT_2 = tf.Variable(tf.random_normal([EMBEDDING_SIZE, VOCAB_SIZE]))
BIASE_2 = tf.Variable(tf.random_normal([VOCAB_SIZE]))
PREDICTION = tf.nn.softmax(tf.add(tf.matmul(Z1, WEIGHT_2), BIASE_2))

# Run Session
SESS = tf.Session()
INIT = tf.global_variables_initializer()

SESS.run(INIT)

LOSS = tf.reduce_mean(-tf.reduce_sum(Y_INPUT * tf.log(PREDICTION), reduction_indices = [1]))

OPTIMIZER = tf.train.GradientDescentOptimizer(0.5).minimize(LOSS)

NUM_ITERATION = 200001

for i in range(NUM_ITERATION):
    SESS.run(OPTIMIZER, feed_dict = {X_INPUT : X_TRAIN, Y_INPUT : Y_TRAIN})
    if i % 1000 == 0:
        print('Iteration', i, ': Loss', SESS.run(LOSS, feed_dict = {X_INPUT : X_TRAIN, Y_INPUT : Y_TRAIN}))

VECTORS = SESS.run(WEIGHT_1 + BIASE_1)

# Data Visualization
MODEL = TSNE(n_components = 2, random_state = 0)
np.set_printoptions(suppress = True)
VECTORS = MODEL.fit_transform(VECTORS)
NORMALIZER = preprocessing.Normalizer()
VECTORS = NORMALIZER.fit_transform(VECTORS, 'l2')

fig, ax = plt.subplots()

for WORD in WORDS:
    print(WORD, VECTORS[WORD2INT[WORD]][1])
    ax.annotate(WORD, (VECTORS[WORD2INT[WORD]][0], VECTORS[WORD2INT[WORD]][1]))

plt.show()
