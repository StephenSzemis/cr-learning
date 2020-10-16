import json
import glob
from nltk.tokenize import TreebankWordTokenizer
from random import shuffle
from bs4 import BeautifulSoup

exclude_chars = '.?!@#$→,()[]-;'

def grab_data():
    dataset = []
    for i, transcript_path in enumerate(glob.iglob('./data/cr_transcripts/*.html')):
        print('Running data grabbing...' + str(i))
        htmlFile = open(transcript_path, "r")
        htmlParser = BeautifulSoup(htmlFile, "html.parser")
        htmlFile.close()

        line_list = [line.text for line in htmlParser.find_all('dd')]

        dataset.append(' '.join(line_list))
    shuffle(dataset)
    return dataset

def tokentime(dataset):
    data = []
    tokenizer = TreebankWordTokenizer()
    for i, sample in enumerate(dataset):
        print('Running tokenization...' + str(i))
        sample = sample.translate(str.maketrans('', '', exclude_chars)).lower().strip()
        data.append(tokenizer.tokenize(sample, return_str=True))
    return data

big_list = tokentime(grab_data())

print("Time to write all the data to a file!")

file_object = open("./analogies/data/tokens.txt", "w")
file_object.write("\n".join(big_list))
file_object.close()