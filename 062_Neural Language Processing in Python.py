# Word2Vec in Python
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Download the tokenizer if necessary
nltk.download('punkt')

# Sample sentences
my_sentences = [
    "we are eager to learn about nlp",
    "neural networks are fun",
    "python is my favourite language"
]

# Tokenize sentences
my_corpus = [word_tokenize(doc) for doc in my_sentences]

# Initialize the Word2Vec model
w2v_model = Word2Vec(sg=0,           # Use CBOW (sg=0)
                     vector_size=100, # Vector size (formerly 'size')
                     alpha=0.025,     # Learning rate
                     min_alpha=0.0001,
                     window=5,        # Window size
                     min_count=1,     # Minimum count to include words (lowered to 1 for this example)
                     sample=0.001,    # Down-sampling rate for frequent words
                     negative=5,      # Number of negative samples
                     workers=3)       # Number of worker threads

# Build the vocabulary
w2v_model.build_vocab(corpus_iterable=my_corpus, update=False)

# Train the Word2Vec model
w2v_model.train(corpus_iterable=my_corpus,
                total_examples=w2v_model.corpus_count,
                epochs=20)

# Check if the word "nlp" is in the vocabulary and get its vector
if "nlp" in w2v_model.wv:
    print(w2v_model.wv["nlp"])
else:
    print("The word 'nlp' is not in the vocabulary.")

from gensim.models import KeyedVectors
import gensim

# Load pre-trained vectors
w2v_model = KeyedVectors.load_word2vec_format("path/to/vectors", binary=True)

# Save the trained model in word2vec format
w2v_model.save_word2vec_format("path/to/model.bin", binary=True)

# Load the saved word2vec model
loaded_model = KeyedVectors.load_word2vec_format("path/to/model.bin", binary=True)

# Example usage
if "example_word" in loaded_model:
    print(loaded_model["example_word"])
else:
    print("The word 'example_word' is not in the vocabulary.")

# Doc2Vec in Python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# Sample corpus
my_corpus = [
    ["we", "are", "eager", "to", "learn", "about", "nlp"],
    ["neural", "networks", "are", "fun"],
    ["python", "is", "my", "favourite", "language"]
]

# Create tagged documents for the Doc2Vec model
tagged_corpus = [TaggedDocument(words=d, tags=["d_" + str(i)]) for i, d in enumerate(my_corpus)]

# Initialize the Doc2Vec model
d2v_model = Doc2Vec(dm=1,             # Distributed Memory (PV-DM)
                    dm_concat=0,       # Use sum of context word vectors
                    dm_mean=1,         # Use mean of context word vectors
                    vector_size=100,   # Dimensionality of the feature vectors
                    window=5,          # The maximum distance between the current and predicted word
                    alpha=0.025,       # The initial learning rate
                    min_alpha=0.0001,  # Learning rate will linearly drop to `min_alpha` as training progresses
                    min_count=1,       # Ignores all words with total frequency lower than this
                    epochs=20,         # Number of epochs to train the model
                    workers=4)         # Number of worker threads to train the model

# Build vocabulary
d2v_model.build_vocab(tagged_corpus)

# Train the model
d2v_model.train(tagged_corpus,
                total_examples=d2v_model.corpus_count,
                epochs=d2v_model.epochs)

# Get the most similar documents to a given document
print(d2v_model.dv.most_similar("d_0", topn=3))

# Compute similarity between two documents
print(d2v_model.dv.similarity("d_0", "d_1"))

# Infer a vector for a new document
new_doc_embedding = d2v_model.infer_vector(["my", "new", "document"])
print(new_doc_embedding)

# Example operations on word vectors (optional)
print(d2v_model.wv.most_similar("nlp", topn=3))
print(d2v_model.wv.similarity("python", "language"))

# FastText in Python
import gensim
from gensim.models.fasttext import FastText
from nltk.tokenize import word_tokenize
import nltk

# Download the tokenizer if necessary
nltk.download('punkt')

# Sample sentences
my_sentences = [
    "we are eager to learn about nlp",
    "neural networks are fun",
    "python is my favourite language"
]

# Tokenize sentences
my_corpus = [word_tokenize(doc) for doc in my_sentences]

# Initialize the FastText model
ft_model = FastText(
    sg=0,              # Use CBOW (sg=0), if sg=1 then Skip-gram is used
    cbow_mean=1,       # Use the mean of the context word vectors
    vector_size=100,   # Dimensionality of the feature vectors
    alpha=0.025,       # Initial learning rate
    min_alpha=0.0001,  # Learning rate will linearly drop to `min_alpha` as training progresses
    window=5,          # The maximum distance between the current and predicted word
    min_count=1,       # Minimum frequency count of words
    sample=0.001,      # Threshold for configuring which higher-frequency words are randomly downsampled
    negative=5,        # If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn
    min_n=3,           # Minimum length of char n-grams to be used for training word representations
    max_n=6            # Maximum length of char n-grams to be used for training word representations
)

# Build vocabulary
ft_model.build_vocab(corpus_iterable=my_corpus)

# Train the FastText model
ft_model.train(
    corpus_iterable=my_corpus,
    total_examples=ft_model.corpus_count,
    epochs=20
)

# Check if words are in the vocabulary and get their vectors
print("example" in ft_model.wv.key_to_index)  # Check if 'example' is in the vocabulary
print(ft_model.wv["example"])                 # Get vector for 'example' if it exists

print("examples" in ft_model.wv.key_to_index) # Check if 'examples' is in the vocabulary
print(ft_model.wv["examples"])                # Get vector for 'examples' if it exists

# Example operations on word vectors
print(ft_model.wv.most_similar("example", topn=3))
print(ft_model.wv.similarity("python", "language"))

'''
Course: Machine Learning and Deep Learning with Python
SoSe 2024
LMU Munich, Department of Statistics
Exercise 6: word2vec / doc2vec / fasttext
'''

import pandas as pd
import numpy as np
import re
import nltk
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models.fasttext import FastText

# %% ------------------------------------------------------------------------------------
# BLOCK 1: Inspecting the data + basic preprocessing
# --------------------------------------------------
print('#' * 50)
print('########## Reading the data ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Import the data using pandas
print('---------- EX-01 ----------')

tweet_data = pd.read_csv('trump.csv')
tweet_data.head()

# %% ------------------------------------------------------------------------------------
# EX02: Extract the tweets to a list of strings
print('---------- EX-02 ----------')

tweets_raw = [tweet for tweet in list(tweet_data.text)]
print(tweets_raw[0])

# %% ------------------------------------------------------------------------------------
# EX03: Convert everything to lowercase
print('---------- EX-03 ----------')

tweets = [doc.lower() for doc in tweets_raw]
print(tweets[0])

# %% ------------------------------------------------------------------------------------
# EX04: Delete url adresses and other unwanted tokens
# (use one list comprehension for deleting urls, one for the others and one for tokenization)
print('---------- EX-04 ----------')

regex_urls = r"https://.*|"
regex_other = r"[\)\(\.\,;:!?\+\-\_\#\'\*\?\$\%\&]"

tweets = [re.sub(regex_urls, "", doc) for doc in tweets]  # url adresses
tweets = [re.sub(regex_other, "", doc) for doc in tweets]  # other unwanted tokens
tweets = [nltk.word_tokenize(doc) for doc in tweets]
print(tweets[0])

# %% ------------------------------------------------------------------------------------
# BLOCK 2: Learning word vector representations
# ---------------------------------------------
print('#' * 50)
print('########## Word2Vec ##########')
print('#' * 50)

# First, we determine the number of CPUs that are available on our machine
# (The more cores are available, the faster we can train our model)

import multiprocessing

cpus = multiprocessing.cpu_count()
print(cpus)

# %% ------------------------------------------------------------------------------------
# EX01: Set up the model (use only the defaults, use all your gpu cores except one)
print('---------- EX-01 ----------')

w2v_model = Word2Vec(workers=cpus - 1)

# %% ------------------------------------------------------------------------------------
# EX02: Build the vocabulary
print('---------- EX-02 ----------')

w2v_model.build_vocab(corpus_iterable=tweets, update=False)

# %% ------------------------------------------------------------------------------------
# EX03: Train the model
print('---------- EX-03 ----------')

w2v_model.train(corpus_iterable=tweets, total_examples=w2v_model.corpus_count, epochs=100)

# %% ------------------------------------------------------------------------------------
# EX04: Now: Explore your model, e.g.
print('---------- EX-04 ----------')

print(w2v_model.wv.most_similar(positive=["germany"]))
print(w2v_model.wv.most_similar(positive=["clinton"]))
print(w2v_model.wv.most_similar(positive=["democrats"]))
print(w2v_model.wv.most_similar(positive=["mexico"]))
print(w2v_model.wv.most_similar(positive=["china"]))
print(w2v_model.wv.most_similar(positive=["mexico", "trade"], negative=["wall"]))

# %% ------------------------------------------------------------------------------------
# EX05: Explore the possibilities the model by e.g. switching from skip-gram to cbow, using concatenation instead of averaging,
# chosing a larger embedding size, more negative examples, etc.
print('---------- EX-05 ----------')

# %% ------------------------------------------------------------------------------------
print('#' * 50)
print('########## Bigrams ##########')
print('#' * 50)
# %% ------------------------------------------------------------------------------------
# EX06: Use gensim.models.phrases in order to form bigrams
# (use min_count=20, threshold=10)
print('---------- EX-06 ----------')

from gensim.models.phrases import Phrases, Phraser

phrases = Phrases(tweets, min_count=20, threshold=10)
bigram = Phraser(phrases)

# %% ------------------------------------------------------------------------------------
# EX07: Display the found bigrams (sorted alphabetically)
print('---------- EX-07 ----------')

sorted_bigrams = sorted(list(bigram.phrasegrams.items()))
print(sorted_bigrams)

# %% ------------------------------------------------------------------------------------
# EX08: Display the found bigrams (sorted alphabetically)
print('---------- EX-08----------')

bigram_tweets = list(bigram[tweets])
print(bigram_tweets[0])

# %% ------------------------------------------------------------------------------------
# EX09: Retrain your model based on the new corpus containing bigrams
print('---------- EX-09 ----------')

bi_model = Word2Vec(workers=cpus - 1)

bi_model.build_vocab(corpus_iterable=bigram_tweets, update=False)
bi_model.train(corpus_iterable=bigram_tweets, total_examples=bi_model.corpus_count, epochs=100)

# %% ------------------------------------------------------------------------------------
# EX10: Select one of the bigrams and compute the cosine similarity with the sum of the
# corresponding vectors from the unigram mode (e.g. "united" and "states" compared to "united_states")
print('---------- EX-10 ----------')

import math

# sum up the vectors of the unigrams
unigrams = w2v_model.wv["united"] + w2v_model.wv["states"]

# extract the vector for the bigram
bigram = bi_model.wv["united_states"]

# calculate the cosine similarity
cos_sim = sum(bigram * unigrams) / (math.sqrt(sum(bigram ** 2)) * math.sqrt(sum(unigrams ** 2)))
print(cos_sim)

# %% ------------------------------------------------------------------------------------
# EX11: Explore the embeddings for the bigrams, e.g.
print('---------- EX-11 ----------')

print(bi_model.wv.most_similar(positive=["united_states"]))
print(bi_model.wv.most_similar(positive=["mueller_report"]))
print(bi_model.wv.most_similar(positive=["north_carolina"]))

# %% ------------------------------------------------------------------------------------
# EX12: Optional task: Run the Phraser again, but this time on the corpus which already contains the bigrams.
# This allows the model to build meaningful trigrams, like e.g. "new_york_times"
print('---------- EX-12 ----------')

phrases_tri = Phrases(bigram_tweets, min_count=20, threshold=10)
trigram = Phraser(phrases_tri)

sorted(list(trigram.phrasegrams.items()))
trigram_tweets = list(trigram[tweets])

# %% ------------------------------------------------------------------------------------
# BLOCK 3: Learning document vector representations
# -------------------------------------------------
print('#' * 50)
print('########## Doc2Vec ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Prepare the data set by transforming every tweet to a TaggedDocument
print('---------- EX-01 ----------')

tagged_tweets = [TaggedDocument(words=d, tags=["doc_" + str(i)]) for i, d in enumerate(tweets)]
print(tagged_tweets[0])

# %% ------------------------------------------------------------------------------------
# EX02: Additional Task: Think about assigning multiple tags to each of the tweets.
# This could be interesting, if we had tweets from different politicians and wanted
# to learn additional representations for their style of tweeting.
# Try to assign a document identifier as well as the label donald_trump to all our tweets
print('---------- EX-02 ----------')

two_tagged_tweets = [TaggedDocument(words=d, tags=["doc_" + str(i), "donald_trump"]) for i, d in enumerate(tweets)]
print(two_tagged_tweets[0])

# %% ------------------------------------------------------------------------------------
# EX03: Set up the model parameters for the Distributed memory model, build the vocab and train it
# (Now again with the corpus which documents are only assigned one tag)
print('---------- EX-03 ----------')

d2v_model = Doc2Vec(dm=1, workers=cpus - 1)

d2v_model.build_vocab(corpus_iterable=tagged_tweets, update=False)
d2v_model.train(corpus_iterable=tagged_tweets, total_examples=d2v_model.corpus_count, epochs=20)

# %% ------------------------------------------------------------------------------------
# EX04: Chose a document and display it as a text
print('---------- EX-04 ----------')

selected_tweet = " ".join(tagged_tweets[10].words)
print(selected_tweet)

# %% ------------------------------------------------------------------------------------
# EX05: Find the IDs of the three most similar tweets to the one you chose
print('---------- EX-05 ----------')

ids_similar = d2v_model.dv.most_similar(["doc_10"], topn=3)
print(ids_similar)

# %% ------------------------------------------------------------------------------------
# EX06: Display them as strings
print('---------- EX-06 ----------')

strings_similar = [" ".join(tweet.words) for tweet in tagged_tweets if tweet.tags[0] in [id[0] for id in ids_similar]]
print(strings_similar)

# %% ------------------------------------------------------------------------------------
# EX07: Compute the cosine similarity to the most similar one of the three tweets
print('---------- EX-07 ----------')

sim = d2v_model.dv.similarity("doc_10", "doc_3177")
print(sim)

# %% ------------------------------------------------------------------------------------
# EX08: Train a Distributed Bag-of-words model
print('---------- EX-08 ----------')

dbow_model = Doc2Vec(dm=0, workers=cpus - 1)

dbow_model.build_vocab(corpus_iterable=tagged_tweets, update=False)
dbow_model.train(corpus_iterable=tagged_tweets, total_examples=dbow_model.corpus_count, epochs=20)

# %% ------------------------------------------------------------------------------------
# EX09: Compare how well the two models were able to learn meaningful word embeddings
# (i.e. extract to most similar words to one pivotal word, e.g. "democrats")
print('---------- EX-09 ----------')

d2v_sim = d2v_model.wv.most_similar(positive=["democrats"])
print(d2v_sim)

dbow_sim = dbow_model.wv.most_similar(positive=["democrats"])
print(dbow_sim)

# %% ------------------------------------------------------------------------------------
# EX10: Now train a second Distributed Bag-of-words model and set the dbow_words-option to 1
# then check the most similar words to your chosen pivotal word again
print('---------- EX-10 ----------')

dbow2_model = Doc2Vec(dm=0, dbow_words=1, workers=cpus - 1)

dbow2_model.build_vocab(corpus_iterable=tagged_tweets, update=False, progress_per=10000)
dbow2_model.train(corpus_iterable=tagged_tweets, total_examples=dbow2_model.corpus_count, epochs=20)

dbow2_sim = dbow2_model.wv.most_similar(positive=["democrats"])
print(dbow_sim)

# %% ------------------------------------------------------------------------------------
# BLOCK 4: Learning subword vector representations
# ------------------------------------------------
print('#' * 50)
print('########## FastText ##########')
print('#' * 50)

# %% ------------------------------------------------------------------------------------
# EX01: Set up the model parameters for the Distributed memory model, build the vocab and train it
# (Now again with the corpus which documents are only assigned one tag)
print('---------- EX-01 ----------')

ft_model = FastText(workers=cpus - 1)

ft_model.build_vocab(corpus_iterable=tweets, update=False)
ft_model.train(corpus_iterable=tweets, total_examples=ft_model.corpus_count, epochs=100)

# %% ------------------------------------------------------------------------------------
# EX02: Check, whether the word "example" does occur in your model's vocabulary
print('---------- EX-02 ----------')

in_vocab = "example" in ft_model.wv.key_to_index
print(in_vocab)

# %% ------------------------------------------------------------------------------------
# EX03: Try to query your word2vec model for a vector representation of this word
print('---------- EX-03 ----------')

w2v_model.wv["example"]

# %% ------------------------------------------------------------------------------------
# EX04: Now try to query your fastText model for a vector representation of this word
print('---------- EX-04 ----------')

vec_example = ft_model.wv["example"]
print(vec_example)

# %% ------------------------------------------------------------------------------------
# EX05: Print the words with the most similar vector representations
print('---------- EX-05 ----------')

words_similar = ft_model.wv.most_similar(positive=["example"])
print(words_similar)

# %% ------------------------------------------------------------------------------------
# EX06: Check, whether the word "democrats" does occur in your model's vocabulary
print('---------- EX-06 ----------')

dem_in_vocab = "democrats" in ft_model.wv.key_to_index
print(dem_in_vocab)

# %% ------------------------------------------------------------------------------------
# EX07: Query your word2vec and you fasttext model for a vector representation of this word
print('---------- EX-07 ----------')

w2v_dems = w2v_model.wv["democrats"]
print(w2v_dems)

ft_dems = ft_model.wv["democrats"]
print(ft_dems)

# %% ------------------------------------------------------------------------------------
# EX08: Print the most word with the most similar vector representations (for the w2v model)
print('---------- EX-08 ----------')

w2v_sim = w2v_model.wv.most_similar(positive=["democrats"])
print(w2v_sim)

# %% ------------------------------------------------------------------------------------
# EX09: Print the most word with the most similar vector representations (for the fasttext model)
print('---------- EX-09 ----------')

ft_sim = ft_model.wv.most_similar(positive=["democrats"])
print(ft_sim)

# %% ------------------------------------------------------------------------------------
# EX10: Do you recognize any systematic differences?
# Explore the possibilities the model by e.g. switching from skip-gram to cbow, trying different
# n-gram ranges, chosing a larger embedding size, etc.
print('---------- EX-10 ----------')
