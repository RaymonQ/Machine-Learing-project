import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def dump_model(model):
    with open('Models/MNB.pickle', 'wb') as output:
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

# ADD CODE MODEL TRAINING HERE:

# see default parameters of multinominal Bayes classifier
multinominalBayes = MultinomialNB()
print(multinominalBayes.get_params())

# fit the model
multinominalBayes.fit(features_train, labels_train)

predicted_classes_train = multinominalBayes.predict(features_train)
predicted_classes_test = multinominalBayes.predict(features_test)

print('The accuracy of MultinominalBayes on the TRAIN set is: ' +
      str(round(accuracy_score(labels_train, predicted_classes_train)*100, 2)) + ' %.')
print('The accuracy of MultinominalBayes on the TEST set is: ' +
      str(round(accuracy_score(labels_test, predicted_classes_test)*100, 2)) + ' %.')
dump_model(multinominalBayes)
# The output for 1000 unigrams is:
# The accuracy of MultinominalBayes on the TRAIN set is: 78.83 %.
# The accuracy of MultinominalBayes on the TEST set is: 74.0 %.
# The output for 300 unigrams is:
# The accuracy of MultinominalBayes on the TRAIN set is: 74.66 %.
# The accuracy of MultinominalBayes on the TEST set is: 70.96 %.

