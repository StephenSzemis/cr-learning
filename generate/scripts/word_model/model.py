import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop


# Get our data
def get_data():
    with open('./generate/data/tokens.txt', 'r') as file:
        tokens = file.read().split()[:3000000]
    print('Grabbed {} tokens from CR transcripts'.format(len(tokens)))
    return tokens

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
    # W_norm = np.zeros(W.shape)
    # d = (np.sum(W ** 2, 1) ** (0.5))
    # W_norm = (W.T / d).T

    return (W, vocab, ivocab)



# Convert to vector of words and train
def train(words, W, vocab, ivocab):
    sentences = []
    next_words = []
    maxlen = 20
    step = 3
    word_vec_size = 60

    for i in range(0, len(words) - maxlen, step):
        sentences.append(words[i: i + maxlen])
        next_words.append(words[i + maxlen])
    # print(next_words)
    
    # Dimensions are (sentences, words, descriptors of words)
    x = np.zeros((len(sentences), maxlen, len(W[0])), dtype=np.float)
    # Dimensions are (sentences, descriptor of next word)
    y = np.zeros((len(sentences), len(W[0])), dtype=np.float)

    for s, sentence in enumerate(sentences):
        for w, word in enumerate(sentence):
            x[s, w] = W[vocab[word]]
        y[s] = W[vocab[next_words[s]]]

    a, b, c = np.shape(x)
    print('We have {} sentences, {} words per sentence, and {} descriptors per word'.format(a, b, c))
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, word_vec_size), return_sequences=True))
    # model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(word_vec_size))
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss='cosine_similarity', optimizer=optimizer)
    model.summary()

    epochs = 6
    batch_size = 16
    model_structure = model.to_json()
    with open('./generate/data/word_lstm_model.json', 'w') as json_file:
        json_file.write(model_structure)
    for i in range(3):
        model.fit(x, y, batch_size=batch_size, epochs=epochs)
        model.save_weights('./generate/data/word_lstm_weights_{}.h5'.format(i+1))


# Profit?
words = get_data()
W, vocab, ivocab = get_vectors()
train(words, W, vocab, ivocab)