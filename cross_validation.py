import tensorflow as tf
import sklearn
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def create_RNN_model(feat_num = 50):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(len(vocab), feat_num))
    model.add(keras.layers.LSTM(feat_num))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def cross_validate_RNN(train_data, train_labels):
    model = KerasClassifier(build_fn=create_RNN_model, verbose=0)

    feat_num_list = [10, 50, 100, 200]
    epochs = [1, 2]
    batches = [32, 64, 128, 256]

    param_grid = dict(feat_num=feat_num_list, epochs=epochs, batch_size=batches)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(train_data, train_labels)

    return grid_result.best_score_, grid_result.best_params_



def create_CNN_model(feat_num = 50, dense_units=250, filter_len = 32, kernel_s = 3):
    model = keras.Sequential()
    model.add(keras.layers.Embedding(len(vocab), feat_num))
    model.add(keras.layers.Convolution1D(filters = filter_len, kernel_size = kernel_s, activation='relu'))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(dense_units, activation=tf.nn.relu))
    model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
    
    model.compile(optimizer=tf.train.AdamOptimizer(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def cross_validate_CNN(train_data, train_labels):
    model = KerasClassifier(build_fn=create_CNN_model, verbose=0)

    dense_units = [100, 150, 200, 250, 300]
    filter_lengths = [16, 32, 64]
    kernel_sizes = [1, 2, 3]
    feat_num_list = [50, 100, 200]
    epochs = [1, 2, 3]
    batches = [32, 64, 128, 256]

    param_grid = dict(kernel_s=kernel_sizes, dense_units=dense_units, filter_len=filter_lengths, feat_num=feat_num_list, epochs=epochs, batch_size=batches)
    grid = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_result = grid.fit(train_data, train_labels)
    
    return grid_result.best_score_, grid_result.best_params_


def cross_validatation(model, parameters):
    grid = GridSearchCV(text_clf, parameters)
    grid_result = grid.fit(train_data, train_labels)

    return grid_result.best_score_, grid_result.best_params_
