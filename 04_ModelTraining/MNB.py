import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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

# ADD CODE MODEL TRAINING HERE:

# see default parameters of multinominal Bayes classifier
multinominalBayes = MultinomialNB()
print(multinominalBayes.get_params())

# fit the model
multinominalBayes.fit(features_train, labels_train)

predicted_classes = multinominalBayes.predict(features_test)

print('The accuracy of MultinominalBayes is: ' + str(round(accuracy_score(labels_test, predicted_classes)*100, 2))
      + ' %.')

