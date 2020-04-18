import pickle
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import time


def dump_model(model):
    with open('Models/RF.pickle', 'wb') as output:
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
    # see default parameters of RF classifier
    rf_default = Rfc(random_state=0)
    print(rf_default.get_params())

    # fit the model
    rf_default.fit(features_train, labels_train)

    predicted_classes_train = rf_default.predict(features_train)
    predicted_classes_test = rf_default.predict(features_test)

    print('The accuracy of the default Random Forest classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train)*100, 2)) + ' %.')

    print('The accuracy of the default Random Forest classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test)*100, 2)) + ' %.')
    dump_model(rf_default)
    # Output for 1000 unigrams:
    # The accuracy of the default Random Forest classifier on the TRAIN set is: 97.95 %.
    # The accuracy of the default Random Forest classifier on the TEST set is: 73.27 %.
    # Output for 300 unigrams:
    # The accuracy of the default Random Forest classifier on the TRAIN set is: 97.67 %.
    # The accuracy of the default Random Forest classifier on the TEST set is: 71.38 %.
if calc_tuned_model:
    # Tuning the RF classifier using first a random search and then a grid search to find the optimal parameters

    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(20, 100, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 4, 8]
    min_samples_leaf = [1, 2, 4]

    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    print(random_grid)

    rfc_model = Rfc(random_state=0)

    # Definition of the random search
    start_time = time.process_time()
    random_search = RandomizedSearchCV(estimator=rfc_model,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=10,
                                       random_state=0)

    # Fit the random search model
    random_search.fit(features_train, labels_train)
    end_time = time.process_time() - start_time
    rfc_best_random = random_search.best_estimator_
    predicted_classes_train = rfc_best_random.predict(features_train)
    predicted_classes_test = rfc_best_random.predict(features_test)
    print("Best parameters found in random search: ")
    print(random_search.best_params_)
    print("\nAccuracy with best parameters:")
    print(random_search.best_score_)
    print("Time used for random search: " + str(round(end_time/60, 2)) + " min.")

    print('The accuracy of the best Random Forest classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')
    print('The accuracy of the best Random Forest classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
    dump_model(rfc_best_random)
    # Output for 1000 unigrams

    # {'n_estimators': [100, 200, 300, 400, 500], 'max_features': ['auto', 'sqrt'],
    # 'max_depth': [20, 40, 60, 80, 100, None], 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]}
    # Fitting 3 folds for each of 32 candidates, totalling 96 fits
    # [Parallel(n_jobs=1)]: Done  96 out of  96 | elapsed:  7.4min finished
    # Best parameters found in random search:
    # {'n_estimators': 500, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 100}
    #
    # Accuracy with best parameters:
    # 0.7418677859391396
    # Time used for random search: 7.43 min.

    # Output for 300 unigrams:
    # Best parameters found in random search:
    # {'n_estimators': 600, 'min_samples_split': 4, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': None}
    #
    # Accuracy with best parameters:
    # 0.7300629590766002
    # Time used for random search: 11.64 min.

if calc_fixed_model:
    # see default parameters of RF classifier
    rf_fixed = Rfc(random_state=0, n_estimators=500, max_features='sqrt', max_depth=100, min_samples_split=2,
                   min_samples_leaf=1)

    rf_fixed = Rfc(random_state=0, n_estimators=600, max_features='auto', max_depth=None,
                   min_samples_split=4, min_samples_leaf=1)

    # fit the model
    rf_fixed.fit(features_train, labels_train)

    predicted_classes_train = rf_fixed.predict(features_train)
    predicted_classes_test = rf_fixed.predict(features_test)

    print('The accuracy of the fixed Random Forest classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train)*100, 2)) + ' %.')

    print('The accuracy of the fixed Random Forest classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test)*100, 2)) + ' %.')
    dump_model(rf_fixed)
    # Output 1
    # The accuracy of the fixed Random Forest classifier on the TRAIN set is: 98.43 %.
    # The accuracy of the fixed Random Forest classifier on the TEST set is: 73.79 %.
    # output randsearch parameters
    # The accuracy of the fixed Random Forest classifier on the TRAIN set is: 98.4 %.
    # The accuracy of the fixed Random Forest classifier on the TEST set is: 73.06 %.
