import pickle
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, ShuffleSplit, GridSearchCV
import numpy as np
import time


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

randomSearch = 0
gridSearch = 1

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

if calc_tuned_model:
    if randomSearch:
        n_neigh = [int(x) for x in np.linspace(start=1, stop=500, num=100)]

        random_grid = {'n_neighbors': n_neigh}

        print(random_grid)
        # First create the base model to tune
        knn_model = Knn()

        # Definition of the random search
        start_time = time.process_time()
        random_search = RandomizedSearchCV(estimator=knn_model,
                                           param_distributions=random_grid,
                                           n_iter=50,
                                           scoring='accuracy',
                                           cv=3,
                                           verbose=10,
                                           random_state=0, n_jobs=3)

        # Fit the random search model
        random_search.fit(features_train, labels_train)
        end_time = time.process_time() - start_time
        knn_best_random = random_search.best_estimator_
        predicted_classes_train = knn_best_random.predict(features_train)
        predicted_classes_test = knn_best_random.predict(features_test)

        print("Best parameters found in random search: ")
        print(random_search.best_params_)
        print("\nAccuracy with best parameters:")
        print(random_search.best_score_)
        print("Time used for random search: " + str(round(end_time / 60, 2)) + " min.")

        print('The accuracy of the best KNN  classifier on the TRAIN set is: ' +
              str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')
        print('The accuracy of the best KNN  classifier on the TEST set is: ' +
              str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
        dump_model(knn_best_random)
        # [Parallel(n_jobs=3)]: Done 150 out of 150 | elapsed: 18.7min finished
        # Best parameters found in random search:
        # {'n_neighbors': 11}
        #
        # Accuracy with best parameters:
        # 0.7589192025183631
        # Time used for random search: 0.16 min.
        # The accuracy of the best KNN  classifier on the TRAIN set is: 80.95 %.
        # The accuracy of the best Knn  classifier on the TEST set is: 75.89 %.

    if gridSearch:

        n_neigh = [int(x) for x in np.linspace(start=3, stop=20, num=18)]

        param_grid = {'n_neighbors': n_neigh}
        print(param_grid)
        # Create a base model
        knn_model = Knn()

        # splits in CV with random state
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=0)

        # Instantiate the grid search model
        start_time = time.process_time()
        grid_search = GridSearchCV(estimator=knn_model,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=10,
                                   n_jobs=3)

        # Fit the grid search to the data
        grid_search.fit(features_train, labels_train)
        end_time = time.process_time() - start_time

        knn_best_grid = grid_search.best_estimator_
        predicted_classes_train = knn_best_grid.predict(features_train)
        predicted_classes_test = knn_best_grid.predict(features_test)

        print("Best parameters found in grid search: ")
        print(grid_search.best_params_)
        print("\nAccuracy with best parameters:")
        print(grid_search.best_score_)
        print("Time used for random search: " + str(round(end_time / 60, 2)) + " min.")

        print('The accuracy of the best SVM  classifier on the TRAIN set is: ' +
              str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')
        print('The accuracy of the best SVM  classifier on the TEST set is: ' +
              str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
        dump_model(knn_best_grid)

        # The output for 1000 unigrams is:

        # [Parallel(n_jobs=3)]: Done  54 out of  54 | elapsed:  6.9min finished
        # [CV] ......... n_neighbors=20, score=0.7352941176470589, total=   8.8s
        # Best parameters found in grid search:
        # {'n_neighbors': 13}
        #
        # Accuracy with best parameters:
        # 0.7485426603073662
        # Time used for random search: 0.06 min.
        # The accuracy of the best SVM  classifier on the TRAIN set is: 80.19 %.
        # The accuracy of the best SVM  classifier on the TEST set is: 75.47 %.

if calc_fixed_model:
    knn_fixed = Knn(n_neighbors=13)

    # fit the model
    knn_fixed.fit(features_train, labels_train)

    predicted_classes_train = knn_fixed.predict(features_train)
    predicted_classes_test = knn_fixed.predict(features_test)

    print('The accuracy of the fixed KNN classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')

    print('The accuracy of the fixed KNN classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
    dump_model(knn_fixed)
