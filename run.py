import numpy as np
import tensorflow as tf
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import sequence
from helpers import create_csv_submission, preprocessData, getTokens

TRAIN_POS = 'data/train_pos_full.txt'
TRAIN_NEG = 'data/train_neg_full.txt'
TEST_DATA = 'data/test_data.txt'

VOCAB = 'tools/vocab.pkl'
PREDICTION_PATH = "best_submission.csv"

with open(VOCAB, 'rb') as f:
    vocab = pickle.load(f)

print("Preprocessing data...")

# Converting data in sequences of tokens        
pos_tok = getTokens(TRAIN_POS, vocab)
neg_tok = getTokens(TRAIN_NEG, vocab)
test_data = getTokens(TEST_DATA, vocab)

# Preparing data for training
train_data, train_labels = preprocessData(pos_tok, neg_tok)

# Add padding to the sequence
max_length = len(max(train_data, key=len))
train_data = sequence.pad_sequences(train_data, maxlen=max_length)
test_data = sequence.pad_sequences(test_data, maxlen=max_length)

print("Model loading...")

# Loading model from hdf5 file
model = load_model("best_model.h5")

print("Generating predictions...")

# Generating predictions
y_pred = model.predict(test_data)

# Creating submission file
y_pred[np.where(y_pred <= 0.5)] = 0
y_pred[np.where(y_pred > 0.5)] = 1
create_csv_submission(range(1,10001), y_pred, PREDICTION_PATH, False)

print("Done!")
