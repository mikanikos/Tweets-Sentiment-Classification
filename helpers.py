import numpy as np
import random
import pickle
import subprocess
import csv
from sklearn.utils import shuffle

# Create submission
def create_csv_submission(ids, y_pred, name, is_normalized):
    
    if not is_normalized:
        y_pred = (y_pred * 2) - 1
    
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


# Building features from text
def getFeatures(fileName):
    # Loading embeddings created before
    embeddings = np.load("tools/embeddings.npy")

    feat_repr = []
    with open('tools/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
        with open(fileName) as file:
            for line in file:
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                if (len(tokens) == 0):
                    tokens = [-1]
                embed_sum = np.zeros(embeddings.shape[1])
                for t in tokens:
                    embed_sum = np.sum([embed_sum, embeddings[t]], axis=0)
                feat_repr.append(embed_sum/len(tokens))
    return np.array(feat_repr)


# Converting words in tokens
def getTokens(fileName, vocab):
    data_tok = []
    with open(fileName) as file:
        for line in file:
            ## Turns all words into their token if in vocab, else -1
            tokens = [vocab.get(t, -1) for t in line.strip().split()]
            ## Filter only existent tokens
            tokens = [t for t in tokens if t >= 0]
            data_tok.append(tokens)         
    return np.array(data_tok)


# Save clean data
def save_data(data, file_name):
    data.to_csv(file_name, header=False, index=False, sep=" ")
    subprocess.call(["sed -i 's/\"//g' " + file_name], shell=True)


# Preprocess data for training
def preprocessData(pos_feat, neg_feat):
    random.seed(123)
    random.shuffle(pos_feat)

    random.seed(123)
    random.shuffle(neg_feat)

    X_tr = np.concatenate((pos_feat, neg_feat))
    y_pos = np.ones(pos_feat.shape[0])
    y_neg = np.zeros(neg_feat.shape[0])
    y_tr = np.concatenate((y_pos, y_neg))

    X_tr, y_tr = shuffle(X_tr, y_tr)

    return X_tr, y_tr


