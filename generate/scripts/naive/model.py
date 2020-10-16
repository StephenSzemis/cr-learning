import glob
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from random import shuffle
from bs4 import BeautifulSoup

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop


exclude_chars = ''

raw_text = "./generate/data/raw_text.txt"
amount_of_chars = 1000000

def grab_data():
    dataset = []
    for i, transcript_path in enumerate(glob.iglob('./data/cr_transcripts/*.html')):
        print('Running data grabbing...' + str(i))
        htmlFile = open(transcript_path, "r")
        htmlParser = BeautifulSoup(htmlFile, "html.parser")
        htmlFile.close()

        line_list = [line.text for line in htmlParser.find_all(['dd', 'dt'])]

        dataset.append('\n'.join(line_list))
    # shuffle(dataset)

    print("Time to write all the data to a file!")

    file_object = open(raw_text, "w")
    file_object.write('\n'.join(dataset))
    file_object.close()

def process():
    file = open(raw_text, 'r')
    text = file.read().lower()
    text = text[:amount_of_chars]
    file.close()
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indicies_chars = dict((i, c) for i, c in enumerate(chars))

    sentences = []
    next_chars = []
    maxlen = 40
    step = 3

    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    
    x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)

    for i, sentences in enumerate(sentences):
        for t, char in enumerate(sentences):
            x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
    optimizer = RMSprop(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    model.summary()

    epochs = 6
    batch_size = 128
    model_structure = model.to_json()
    with open('./generate/data/char_lstm_model.json', 'w') as json_file:
        json_file.write(model_structure)
    for i in range(3):
        model.fit(x, y, batch_size=batch_size, epochs=epochs)
        model.save_weights('./generate/data/char_lstm_weights_{}.h5'.format(i+1))


#grab_data()
process()