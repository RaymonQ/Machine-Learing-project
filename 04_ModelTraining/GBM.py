import pickle
from sklearn.ensemble import GradientBoostingClassifier as Gbc
from sklearn.metrics import accuracy_score


def dump_model(model):
    with open('Models/GBM.pickle', 'wb') as output:
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
    # see default parameters of GBM classifier
    gbc_default = Gbc()
    print(gbc_default.get_params())

    # fit the model
    gbc_default.fit(features_train, labels_train)

    predicted_classes_train = gbc_default.predict(features_train)
    predicted_classes_test = gbc_default.predict(features_test)

    print('The accuracy of the default GBM classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train)*100, 2)) + ' %.')
    print('The accuracy of the default GBM classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test)*100, 2)) + ' %.')

    # The output for 1000 unigrams (took an awful lot of time) is:
    # The accuracy of the default GBM classifier on the TRAIN set is: 96.56 %.
    # The accuracy of the default GBM classifier on the TEST set is: 75.16 %.
    # The output for 300 unigrams is:
    # The accuracy of the default GBM classifier on the TRAIN set is: 96.07 %.
    # The accuracy of the default GBM classifier on the TEST set is: 72.43 %.
    dump_model(gbc_default)

if calc_fixed_model:
    # some parameters of GBM classifier from the internet (towardsDataScience)
    gbc_fixed = Gbc(learning_rate=0.1, max_depth=15, max_features='sqrt', min_samples_leaf=2, min_samples_split=50,
                    n_estimators=800, subsample=1.0)

    # fit the model
    gbc_fixed.fit(features_train, labels_train)

    predicted_classes_train = gbc_fixed.predict(features_train)
    predicted_classes_test = gbc_fixed.predict(features_test)

    print('The accuracy of the fixed GBM classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')
    print('The accuracy of the fixed GBM classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
    dump_model(gbc_fixed)



