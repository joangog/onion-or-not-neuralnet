import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # hide TensorFlow warnings
import timeit # to calculate run time
start = timeit.default_timer()

from math import log
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from keras import optimizers
from keras import metrics
from keras.callbacks.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense

# aux functions
def calc_tf(term,document_terms):
    tf=0
    for document_term in document_terms:
        if document_term == term:
            tf+=1
    return tf

def calc_idf(term,all_document_terms):
    n=0 # number of docs that have the term
    N=len(all_document_terms) # number of docs
    for document_terms in all_document_terms:
        if term in set(document_terms):
            n+=1
    if n != 0:
        return log(N/n)
    else:
        return None # n=0 should never be possible because if it has been added to the list of terms it exists in at least one doc

# import csv
data = pd.read_csv('onion-or-not.csv')

# preprocessing
all_terms = set() # all terms
all_article_terms = [] # terms per article
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))
for article_title in data['text']:
    # separate text terms
    article_terms = nltk.word_tokenize(article_title)
    new_article_terms = []
    for term in article_terms:
        #uncapitalize
        term = term.lower()
        # stemming
        term = stemmer.stem(term)
        # filter stop words
        if term not in stop_words:
            new_article_terms.append(term)
            all_terms.add(term)
    all_article_terms.append(new_article_terms)

# create dataset
dataset = pd.DataFrame(0, index=range(len(data)), columns=list(all_terms)+['class_label']) # init with zeros
for article,article_terms in enumerate(all_article_terms): # for every article
    for term in article_terms: # for every article term calculate its tfidf
        tf = calc_tf(term, article_terms)
        idf = calc_idf(term,all_article_terms)
        dataset.at[article,term] = tf*idf # add the tfidf for the term of the article in the dataset
    dataset.at[article,'class_label'] = data.at[article,'label'] # add the class label to the dataset

# save dataset
dataset.to_csv(path_or_buf='dataset.csv',index=False)

stop = timeit.default_timer()
print(f'Runtime: {int((stop - start)/60)} minutes')