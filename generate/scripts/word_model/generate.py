import numpy as np
from keras.models import model_from_json
from keras.models import Sequential
import random
import numpy as np
import sys

maxlen = 20
step = 3
word_vec_size = 60

json_file = open('./generate/data/word_lstm_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights('./generate/data/word_lstm_weights_1.h5')


# Vector list
def get_vectors():
    vocab_file = "./generate/data/vocab.txt"
    vectors_file = "./generate/data/vectors.txt"
    with open(vocab_file, 'r') as f:
        words = [x.rstrip().split(' ')[0] for x in f.readlines()]
    with open(vectors_file, 'r') as f:
        vectors = {}
        for line in f:
            vals = line.rstrip().split(' ')
            vectors[vals[0]] = [float(x) for x in vals[1:]]

    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T

    return (W, vocab, ivocab)

# Get our data
def get_data():
    with open('./generate/data/tokens.txt', 'r') as file:
        tokens = file.read().split()
    print('Grabbed {} tokens from CR transcripts'.format(len(tokens)))
    return tokens

def find_word(v, W, vocab, ivocab):
    d = (np.sum(v ** 2,) ** (0.5))
    vec_norm = (v.T / d).T
    # vec_norm = v

    dist = np.dot(W, vec_norm.T)
    a = np.argsort(-dist)[0]
    term = ivocab[a]
    return term


print('Predicting text:')
generated = ''
words = get_data()
W, vocab, ivocab = get_vectors()


start_index = random.randint(0, len(words) - maxlen - 1)

sentence = words[start_index: start_index + maxlen]
generated += ' '.join(sentence)
sys.stdout.write(generated)
for i in range(100):
    x = np.zeros((1, maxlen, word_vec_size))
    for w, word in enumerate(sentence):
        x[0, w] = W[vocab[word]]
    pred = model.predict(x, verbose=0)[0]
    next_word = find_word(pred, W, vocab, ivocab)
    generated += next_word
    sentence = sentence[1:] + [next_word]
    sys.stdout.write(' ' + next_word)
    sys.stdout.flush()
print()