import pandas as pd
from preprocessing.preprocessing import cleandata
from preprocessing.vectorization import get_counts, del_words, create_vocab, encode_sentences

df = pd.read_json('/data/Digital_music_5.json', lines = True)

df1 = cleandata(df)

item_counts = get_counts(df1, 'reviewText_item')
user_counts = get_counts(df1, 'reviewText_user')

item_counts = del_words(item_counts)
user_counts = del_words(user_counts)

item_words = create_vocab(item_counts)
user_words = create_vocab(user_counts)

df1['encoded_item'] = df1['reviewText_item'].apply(lambda x: np.array(encode_sentence(x, item_words[1])))
df1['encoded_user']=df1['reviewText_user'].apply(lambda x: np.array(encode_sentence(x, user_words[1])))
