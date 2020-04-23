from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Embedding
from keras.layers.embeddings import Embedding
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ShuffleSplit
import time
import numpy as np


def make_dense(lables, cats):
    labels_dense = np.zeros((lables.shape[0], cats))
    lables = lables.values
    for i in range(labels_dense.shape[0]):
        labels_dense[i, lables[i]] = 1
    return labels_dense


def dump_model(model):
    with open('Models/SVM.pickle', 'wb') as output:
        pickle.dump(model, output)
    print('\nmodel dumped!')


# specify your directory where the project is here
# path Yannick:
path_project = "/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/"

# importing objects from folder data in feature engineering
path_data = path_project + '03_FeatureEngineering/Data/'

with open(path_data + 'df_train.pickle', 'rb') as data:
    df_train = pickle.load(data)
with open(path_data + 'df_test.pickle', 'rb') as data:
    df_test = pickle.load(data)
with open(path_data + 'features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)
with open(path_data + 'features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)
with open(path_data + 'labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)
with open(path_data + 'labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)
with open(path_data + 'tfidf_custom.pickle', 'rb') as data:
    tfidf_custom = pickle.load(data)

EmbeddingDim = 100
vocab_size = features_train.shape[1]
print(vocab_size)
max_length = features_train.shape[1]
print(max_length)
numlabels = 10

sql_model = Sequential()
sql_model.add(Embedding(vocab_size, EmbeddingDim))
sql_model.add(GRU(units=32, dropout=.2, recurrent_dropout=.2))
sql_model.add(Dense(numlabels, activation='softmax'))

sql_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(sql_model.summary())

num_epochs = 10
batch_size = 128
history = sql_model.fit(features_train, make_dense(labels_train, numlabels), batch_size=batch_size, epochs=num_epochs,
                        verbose=2, validation_split=.2)

score, acc = sql_model.evaluate(features_test, make_dense(labels_test, numlabels), batch_size=batch_size, verbose=2)

print(acc)


