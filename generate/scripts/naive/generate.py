from keras.models import model_from_json
from keras.models import Sequential
import random
import numpy as np
import sys

raw_text = "./generate/data/raw_text.txt"
amount_of_chars = 100000
maxlen = 40
step = 3

json_file = open('./generate/data/char_lstm_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


model.load_weights('./generate/data/char_lstm_weights_3.h5')

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

file = open(raw_text, 'r')
text = file.read().lower()
text = text[:amount_of_chars]
file.close()
chars = sorted(list(set(text)))

char_indices = dict((c, i) for i, c in enumerate(chars))
indicies_chars = dict((i, c) for i, c in enumerate(chars))

start_index = random.randint(0, len(text) - maxlen - 1)
for diversity in [0.2, 0.5, 0.7]:
    print()
    print('Predicting text:')
    generated = ''
    sentence = text[start_index: start_index + maxlen]
    generated += sentence
    sys.stdout.write(generated)
    for i in range(400):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indicies_chars[next_index]
        generated += next_char
        sentence = sentence[1:] + next_char
        sys.stdout.write(next_char)
        sys.stdout.flush()
    print()
