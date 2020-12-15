from __future__ import unicode_literals, print_function, division
from collections import Counter
import re
import string
import spacy


tok = spacy.load('en_core_web_sm')
def tokenize (text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]') # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tok.tokenizer(nopunct)]

def get_counts(df, colname):
  counts = Counter()
  for index, row in df.iterrows():
    counts.update(tokenize(row[colname]))
  print(counts)
  return counts

def del_words(counts):
  print("num_words before:",len(counts.keys()))
  for word in list(counts):
    if counts[word] < 2:
      del counts[word]
  print("num_words after:",len(counts.keys()))
  return counts

def create_vocab(counts):
  vocab2index = {"":0, "UNK":1}
  words = ["", "UNK"]
  for word in counts:
    vocab2index[word] = len(words)
    words.append(word)
  return words, vocab2index

def encode_sentence(text, vocab2index, N=70):
    tokenized = tokenize(text)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length
