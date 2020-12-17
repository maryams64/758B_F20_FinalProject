from __future__ import unicode_literals, print_function, division
import numpy as np
from collections import Counter
import re
import string
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


tok = spacy.load('en_core_web_sm')


def tokens(text):
    return word_tokenize(text)

def counter(df, colname):
  counts = Counter()
  for index, row in df.iterrows():
        counts.update(tokens(row['reviewText_item']))
  return counts

def delete(counts):
  for word in list(counts):
      if counts[word] in stop_words or counts[word] < 3:
          del counts[word]
  return counts

def get_vocab_size(counts):
  vocab2index = {"":0, "UNK":1}
  words = ["", "UNK"]
  for word in counts:
    vocab2index[word] = len(words)
    words.append(word)
  return words, vocab2index

def encode(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length
