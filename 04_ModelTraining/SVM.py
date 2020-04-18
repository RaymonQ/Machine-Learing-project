import pickle
from sklearn.svm import SVC
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

# see default parameters of SVC classifier (randomstate should be the same for hypertuning the parameters
# as when we did the splitting in feature_engineering.py
svc_default = SVC(random_state=0)
print(svc_default.get_params())

# model with tuned paras (for another data set of course:) from the internet (towardsdatascience link)
svc_tuned = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                gamma='auto', kernel='linear', max_iter=-1, probability=True, random_state=8, shrinking=True,
                tol=0.001, verbose=False)
# fit the models
svc_default.fit(features_train, labels_train)
svc_tuned.fit(features_train, labels_train)

predicted_classes_default_train = svc_default.predict(features_train)
predicted_classes_tuned_train = svc_tuned.predict(features_train)
predicted_classes_default_test = svc_default.predict(features_test)
predicted_classes_tuned_test = svc_tuned.predict(features_test)


print('The accuracy of default SVC on the TRAIN set is: ' +
      str(round(accuracy_score(labels_train, predicted_classes_default_train)*100, 2)) + ' %.')
print('The accuracy of default SVC on the TEST is: ' +
      str(round(accuracy_score(labels_test, predicted_classes_default_test)*100, 2)) + ' %.')
print('The accuracy of tuned SVC on the TRAIN set is: ' +
      str(round(accuracy_score(labels_train, predicted_classes_tuned_train)*100, 2)) + ' %.')
print('The accuracy of tuned SVC on the TEST is: ' +
      str(round(accuracy_score(labels_test, predicted_classes_tuned_test)*100, 2)) + ' %.')

# Output for 1500 unigrams:
# The accuracy of default SVC on the TRAIN set is: 34.97 %.
# The accuracy of default SVC on the TEST is: 35.64 %.
# The accuracy of tuned SVC on the TRAIN set is: 34.97 %.
# The accuracy of tuned SVC on the TEST is: 35.64 %.
