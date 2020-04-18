import pickle
from sklearn.ensemble import RandomForestClassifier as Rfc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import time

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

calc_default_model = True
calc_tuned_model = False

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

    # Output for 1500 unigrams:
    # The accuracy of the default Random Forest classifier on the TRAIN set is: 55.56 %.
    # The accuracy of the default Random Forest classifier on the TEST set is: 42.24 %.

if calc_tuned_model:
    # Tuning the RF classifier using first a random search and then a grid search to find the optimal parameters

    n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(20, 100, num=5)]
    max_depth.append(None)
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 4]
    # # lazy para
    # n_estimators = [300]
    # max_features = ['auto']
    # max_depth = [30, None]
    # min_samples_split = [5]
    # min_samples_leaf = [2, 4]
    # construct grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    print(random_grid)

    rfc_model = Rfc(random_state=0)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=rfc_model,
                                       param_distributions=random_grid,
                                       n_iter=50,
                                       scoring='accuracy',
                                       cv=3,
                                       verbose=1,
                                       random_state=0)

    # Fit the random search model
    start_time = time.process_time()
    random_search.fit(features_train, labels_train)
    end_time = time.process_time() - start_time

    print("Best parameters found in random search: ")
    print(random_search.best_params_)
    print("\nAccuracy with best parameters:")
    print(random_search.best_score_)
    print("Time used for random search: " + str(round(end_time/60, 2)) + " s.")

    # Output for 1500 unigrams
    # Fitting 3 folds for each of 50 candidates, totalling 150 fits
    # [Parallel(n_jobs=1)]: Done 150 out of 150 | elapsed: 17.8 min finished
    # Best parameters found in random search:
    # {'n_estimators': 400, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 20}
    # Accuracy with best parameters:0.3496852046169989
    # Time used for random search: 17.54 s.
