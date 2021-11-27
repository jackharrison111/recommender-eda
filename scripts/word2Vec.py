import gensim
import time
import pandas as pd
import ast
import numpy as np

import gensim.downloader
from gensim.parsing.preprocessing import remove_stopwords


# Show all available models in gensim-data
print(list(gensim.downloader.info()['models'].keys()))

start = time.perf_counter()
print("Loading model...")
# Load Google's pre-trained Word2Vec model.
#model = gensim.models.KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
model = gensim.downloader.load('word2vec-google-news-300')
#glove_vectors = gensim.downloader.load('glove-twitter-25')
end = time.perf_counter()
print(f"Model loaded. \nTime taken: {end-start} second(s).")


data = pd.read_csv('./data/test_postprocessed_data.csv')
data['token_list'] = data['tokens'].apply(lambda x: ast.literal_eval(x))
#print(data.head())
dataset = []
pad_length = 100
token_list = data['token_list']
for test in token_list:
    embedding_list = []
    
    test = remove_stopwords(test)
    for word in test:
        try:
            vec = model[word]
            embedding_list.append(model[word])
        except:
            print(word, " is not in the corpus.")
    while len(embedding_list) < pad_length:
        embedding_list.append(np.zeros((300,)))
    embedding_list = np.array(embedding_list)
    #print(embedding_list)
    print(embedding_list.shape)
    dataset.append(embedding_list)
    emb_time = time.perf_counter()
    
print(emb_time - end, f" seconds taken to embed {len(dataset)} sentences.")
dataset = np.array(dataset)
print(dataset.shape)

