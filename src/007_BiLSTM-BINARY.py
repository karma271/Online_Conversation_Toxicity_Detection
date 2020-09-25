"""
Description: This script is used to train BiLSTM-Binary Classifier for Toxicity Classification
Author: Prasoon Karmacharya
Last update: 09/05/2020
Note: Same script is used two different embedding layers, GloVe and FastText
"""
# Imports:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import re
import seaborn as sns
import time


import tensorflow as tf
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Embedding, Input, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D,GlobalAveragePooling1D, MaxPooling1D, SpatialDropout1D
from keras.models import Model
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle

script_start = time.time()

# Import Data
data_folder = "../assets/data/jigsaw_data"

print('Loading Neutral toxic Data')
dev = pd.read_csv(data_folder + "/neutral_toxic_data.csv")
print('Done Neutral toxic  Data')

required_cols = ['id', 'comment_text', 'cleaned_comment_text', 'neutral']
# toxicity_classes = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

dev = dev[required_cols]
print(dev.head())

neutral_data_proportion = 0.6
toxic_data_proportion = 0.4
orig_neutral_count= dev.neutral.value_counts()[1]
orig_toxic_count = dev.neutral.value_counts()[0]
print(f"Downsample Netural Data to: {int(orig_neutral_count - (orig_neutral_count-(neutral_data_proportion/toxic_data_proportion)*orig_toxic_count))}")
print(f"i.e, drop {int(orig_neutral_count-(neutral_data_proportion/toxic_data_proportion)*orig_toxic_count)} Neutral Data")


dev = shuffle(dev)
dev = dev.drop(dev[dev["neutral"]==1].sample(119008).index)
print("Distribution after dropping neutral records")
print(dev.neutral.value_counts())
print(dev.neutral.value_counts(normalize=True))


# Original
X_train_dev = dev["cleaned_comment_text"]
y_train_dev = dev["neutral"]

print(X_train_dev.shape, y_train_dev.shape)


# Constants
max_features = 100_000      # no. of worlds to embed, rest will be ignored
max_text_length = 400       # max. length of comments, shorter comments will be padded 
embed_size = 300            # work embedding size


# Tokeniztion
print("Tokenizing ...")

tokenize_start = time.time()

tokenizer = text.Tokenizer(num_words = max_features, lower = True)   

# fit the tokenizer to X_train_dev
tokenizer.fit_on_texts(list(X_train_dev))


# convert tokenized text to list of sequences of numbers
X_train_dev_tokenized = tokenizer.texts_to_sequences(X_train_dev)

# pad each of the equence to max_text_length
X_train_dev_tokenized_padded = sequence.pad_sequences(X_train_dev_tokenized, maxlen=max_text_length)

tokenize_end = time.time()

print(f"Time to tokenize:{tokenize_end - tokenize_start} seconds.")


#GLoVe

print("Starting GloVe embedding")

print('\t Loading GloVe Embeddings')

embedding_folder = "../assets/embeddings"
EMBEDDING_FILE = embedding_folder + "/glove.840B.300d.txt"
print('\t Done Loading GloVe Embeddings')

print('\n \t Started Embedding')
embedding_start = time.time()
embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

word_index = tokenizer.word_index
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_end = time.time()
print(f'\t Found {len(embeddings_index)} word vectors.')
print(f"Embedding completed:{embedding_end - embedding_start} seconds")
print(f"Embedding matrix:{embedding_matrix.shape}")

print(embedding_matrix.shape)


# # FastText

# print("Starting FastText embedding")

# print('\t Loading FastText Embeddings')

# embedding_folder = "../assets/embeddings"
# EMBEDDING_FILE = embedding_folder + "/wiki.simple.vec"

# print('\t Done Loading FastText Embeddings')

# print('\n \t Started Embedding')
# embedding_start = time.time()
# embeddings_index = {}
# with open(EMBEDDING_FILE, encoding='utf8') as f:
#     for line in f:
#         values = line.rstrip().rsplit(' ')
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs

# word_index = tokenizer.word_index
# num_words = min(max_features, len(word_index) + 1)
# embedding_matrix = np.zeros((num_words, embed_size))
# for word, i in word_index.items():
#     if i >= max_features:
#         continue
    
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[i] = embedding_vector

# embedding_end = time.time()
# print(f'\t Found {len(embeddings_index)} word vectors.')
# print(f"Embedding completed:{embedding_end - embedding_start} seconds")
# print(f"Embedding matrix:{embedding_matrix.shape}")


# embedding_matrix.shape

# Modeling

# Train-Validation Split
X_train, X_test, y_train, y_test = train_test_split(X_train_dev_tokenized_padded, y_train_dev, train_size=0.20, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Model Evaluation
class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

# To save the best model with call back          
model_path = "../models/BiLSTM_BINARY/"
model_name = "01_BiLSTM_Glove_BINARY"
# model_name = "02_BiLSTM_FT_BINARY"

best_model_path = model_path + model_name + "_best.h5"


# Callbacks initializer
check_point = ModelCheckpoint(best_model_path, monitor = "val_recall", verbose = 1, 
                              save_best_only = True, mode = "max")
roc_auc = RocAucEvaluation(validation_data=(X_test, y_test), interval=1)
early_stop = EarlyStopping(monitor= "val_loss", mode = "min", patience=8)

# Other metrics 
# (https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model)

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

# # To solve class-imbalance
# def calculating_class_weights(y_true):
#     from sklearn.utils.class_weight import compute_class_weight
#     number_dim = np.shape(y_true)[1]
#     weights = np.empty([number_dim, 2])
#     for i in range(number_dim):
#         weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
#     return weights


# class_weights = calculating_class_weights(y_train)
# class_weights

# Model Architecture
inputs = Input(shape=(max_text_length,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable = False)(inputs)
x = SpatialDropout1D(0.35)(x)
x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1, kernel_initializer='he_normal'))(x)
x = Conv1D(64, kernel_size = 3, padding = "valid", kernel_initializer = "he_normal")(x)
x = GlobalMaxPooling1D()(x)
x = Dense(60, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(1, activation="sigmoid")(x) 

model_BiLSTM = Model(inputs=inputs, outputs=outputs)


model_BiLSTM.compile(loss='binary_crossentropy', optimizer='adam', 
                     metrics=['accuracy', tf.keras.metrics.Recall(), f1_m, precision_m, tf.keras.metrics.AUC() ]
              )
print(model_BiLSTM.summary())


# Model Fit

print("Starting Modeling")
model = model_BiLSTM
start_model = time.time()
batch_size = 32
epochs = 20
history = model.fit(X_train, y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          validation_data = (X_test, y_test),
          verbose=1,
          callbacks=[roc_auc, check_point, early_stop]
        )
end_model = time.time()

print(f"Completed modeling in {(end_model - start_model)/60} minutes.")


# Save Tokenizer and Model

model_path = "../models/BiLSTM_BINARY/"
model_name = "01_BiLSTM_Glove_BINARY"
# model_name = "02_BiLSTM_FT_BINARY"

best_model_path = model_path + model_name + "_best.h5"

# save tokenizer
with open(f"{model_path}{model_name}_tokenizer.pickle", 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# as json
model_json = model.to_json()
with open(model_path + f"{model_name}.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize to HDF5
model.save(model_path + f"{model_name}.h5")


# model weights to HDF5
model.save_weights(model_path + f"{model_name}_weights.h5")
