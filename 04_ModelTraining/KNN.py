import pickle
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.metrics import accuracy_score


def dump_model(model):
    with open('Models/KNN.pickle', 'wb') as output:
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
calc_default_model = 0
calc_tuned_model = 0
calc_fixed_model = 1

if calc_default_model:
    # see default parameters of KNN classifier
    knn_default = Knn()
    print(knn_default.get_params())

    # fit the model
    knn_default.fit(features_train, labels_train)

    predicted_classes_train = knn_default.predict(features_train)
    predicted_classes_test = knn_default.predict(features_test)

    print('The accuracy of the default KNN classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train)*100, 2)) + ' %.')

    print('The accuracy of the default KNN classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test)*100, 2)) + ' %.')
    dump_model(knn_default)

if calc_fixed_model:
    knn_fixed = Knn(n_neighbors=6)

    # fit the model
    knn_fixed.fit(features_train, labels_train)

    predicted_classes_train = knn_fixed.predict(features_train)
    predicted_classes_test = knn_fixed.predict(features_test)

    print('The accuracy of the fixed KNN classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')

    print('The accuracy of the fixed KNN classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
    dump_model(knn_fixed)


# The output for 1000 unigrams is:
# The accuracy of the default KNN classifier on the TRAIN set is: 84.5 %.
# The accuracy of the default KNN classifier on the TEST set is: 75.89 %.
# The output for 300 unigrams is:
# The accuracy of the default KNN classifier on the TRAIN set is: 82.21 %.
# The accuracy of the default KNN classifier on the TEST set is: 73.17 %.
