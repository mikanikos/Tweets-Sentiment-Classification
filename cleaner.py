import pandas as pd
import numpy
from nltk.corpus import stopwords
from textblob import TextBlob, Word
import csv
import string
import subprocess

CLEAN_DATA_POS = 'train_pos_full_clean.txt'
CLEAN_DATA_NEG = 'train_neg_full_clean.txt'
CLEAN_TEST = 'test_data_clean.txt'

# Cleaning function
def clean_data(file, drop_dup = True):
    
    data = pd.read_csv(file, header=None, delimiter="\n", names=["tweet"])
    
    if drop_dup:
        # Removing duplicates
        data.drop_duplicates(inplace=True)
    
    # Putting everything to lower case
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    
    # Removing user tags and possible html tags
    data['tweet'] = data['tweet'].str.replace('<.*?>','')
    
    # Removing urls and references to other people 
    data['tweet'] = data['tweet'].str.replace('@\w+','')
    data['tweet'] = data['tweet'].str.replace('http.?://[^\s]+[\s]?','')
    
    # Removing punctuation and symbols
    data['tweet'] = data['tweet'].str.replace('[^\w\s]', '')
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in string.punctuation))
    
    # Removing non alphabetical characters
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x.isalpha()))
    
    # Removing characters shorter than 2
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if len(x) > 1))
    
    # Removing stopwords by using NLTK list
    sw = stopwords.words('english')
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    
    # Removing digits
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
    
    # Removing words that appear less than 5
    word_freq = pd.Series(' '.join(data['tweet']).split()).value_counts()
    less_freq = word_freq[word_freq < 5]
    data['tweet'] = data['tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
    
    # Removing multiple spaces
    data['tweet'] = data['tweet'].apply(lambda x: x.strip())
    data['tweet'] = data['tweet'].str.replace(' +',' ')

    # Spelling correction
    #data['tweet'] = data['tweet'].apply(lambda x: str(TextBlob(x).correct()))
    
    # Lemmatization
    data['tweet'] = data['tweet'].apply(lambda x: " ".join([Word(w).lemmatize() for w in x.split()]))
    
    if drop_dup:
        # Removing duplicates again
        data.drop_duplicates(inplace=True)
    
    return data
