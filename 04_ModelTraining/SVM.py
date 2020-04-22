import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ShuffleSplit
import time


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

# ADD CODE MODEL TRAINING HERE:
calc_default_model = 0
calc_tuned_model = 0
calc_fixed_model = 1
random_search = 0
grid_search = 0

if calc_default_model:
    # see default parameters of SVC classifier (randomstate should be the same for hypertuning the parameters
    # as when we did the splitting in feature_engineering.py
    svc_default = SVC(random_state=0)
    print(svc_default.get_params())

    svc_default.fit(features_train, labels_train)
    predicted_classes_default_train = svc_default.predict(features_train)
    predicted_classes_default_test = svc_default.predict(features_test)

    print('The accuracy of default SVC on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_default_train)*100, 2)) + ' %.')
    print('The accuracy of default SVC on the TEST is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_default_test)*100, 2)) + ' %.')
    dump_model(svc_default)

if calc_tuned_model:
    if random_search:
        # Tuning the SVM classifier using first a random search and then a grid search to find the optimal parameters
        C = [.0001, .001, .01]
        gamma = [.0001, .001, .01, .1, 1, 10, 100]
        degree = [1, 2, 3, 4, 5]
        kernel = ['linear', 'rbf', 'poly']
        probability = [True]

        random_grid = {'C': C,
                       'kernel': kernel,
                       'gamma': gamma,
                       'degree': degree,
                       'probability': probability
                       }
        print(random_grid)
        # First create the base model to tune
        svc_model = SVC(random_state=0)

        # Definition of the random search
        start_time = time.process_time()
        random_search = RandomizedSearchCV(estimator=svc_model,
                                           param_distributions=random_grid,
                                           n_iter=50,
                                           scoring='accuracy',
                                           cv=3,
                                           verbose=10,
                                           random_state=0)

        # Fit the random search model
        random_search.fit(features_train, labels_train)
        end_time = time.process_time() - start_time
        svm_best_random = random_search.best_estimator_
        predicted_classes_train = svm_best_random.predict(features_train)
        predicted_classes_test = svm_best_random.predict(features_test)

        print("Best parameters found in random search: ")
        print(random_search.best_params_)
        print("\nAccuracy with best parameters:")
        print(random_search.best_score_)
        print("Time used for random search: " + str(round(end_time / 60, 2)) + " min.")

        print('The accuracy of the best SVM  classifier on the TRAIN set is: ' +
              str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')
        print('The accuracy of the best SVM  classifier on the TEST set is: ' +
              str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
        dump_model(svm_best_random)
        # Output of random Search for 300 features:

        # [Parallel(n_jobs=1)]: Done 150 out of 150 | elapsed: 47.5min finished
        # Best parameters found in random search:
        # {'probability': True, 'kernel': 'poly', 'gamma': 100, 'degree': 1, 'C': 0.01}
        #
        # Accuracy with best parameters:
        # 0.75
        # Time used for random search: 46.85 min.
        # The accuracy of the best SVM  classifier on the TRAIN set is: 84.89 %.
        # The accuracy of the best SVM  classifier on the TEST set is: 74.32 %.

    if grid_search:
        # parameter grid based on output random search
        C = [.001, .01, .1]
        degree = [1, 2, 3, 5]
        gamma = [10, 100, 200]
        probability = [True]

        param_grid = [
            {'C': C, 'kernel': ['linear'], 'probability': probability},
            {'C': C, 'kernel': ['poly'], 'degree': degree, 'probability': probability},
            {'C': C, 'kernel': ['rbf'], 'gamma': gamma, 'probability': probability}
        ]

        # Create a base model
        svc_model = SVC(random_state=0)

        # splits in CV with random state
        cv_sets = ShuffleSplit(n_splits=3, test_size=.33, random_state=0)

        # Instantiate the grid search model
        start_time = time.process_time()
        grid_search = GridSearchCV(estimator=svc_model,
                                   param_grid=param_grid,
                                   scoring='accuracy',
                                   cv=cv_sets,
                                   verbose=10)

        # Fit the grid search to the data
        grid_search.fit(features_train, labels_train)
        end_time = time.process_time() - start_time

        svm_best_grid = grid_search.best_estimator_
        predicted_classes_train = svm_best_grid.predict(features_train)
        predicted_classes_test = svm_best_grid.predict(features_test)

        print("Best parameters found in grid search: ")
        print(grid_search.best_params_)
        print("\nAccuracy with best parameters:")
        print(grid_search.best_score_)
        print("Time used for random search: " + str(round(end_time / 60, 2)) + " min.")

        print('The accuracy of the best SVM  classifier on the TRAIN set is: ' +
              str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')
        print('The accuracy of the best SVM  classifier on the TEST set is: ' +
              str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
        dump_model(svm_best_grid)
        # Best parameters found in grid search:
        # {'C': 0.1, 'kernel': 'linear', 'probability': True}
        #
        # Accuracy with best parameters:
        # 0.6958134605193429
        # Time used for random search: 26.37 min.
        # The accuracy of the best SVM  classifier on the TRAIN set is: 73.5 %.
        # The accuracy of the best SVM  classifier on the TEST set is: 72.43 %.

if calc_fixed_model:
    # model with tuned paras (for another data set of course:) from the internet (towardsdatascience link)
    svc_fixed = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3,
                    gamma='auto', kernel='linear', max_iter=-1, probability=True, random_state=0, shrinking=True,
                    tol=0.001, verbose=False)
    svc_fixed = SVC(C=0.01, degree=1, gamma=100, kernel='poly', probability=True, random_state=0)
    # fit the models
    svc_fixed.fit(features_train, labels_train)

    predicted_classes_fixed_train = svc_fixed.predict(features_train)
    predicted_classes_fixed_test = svc_fixed.predict(features_test)

    print('The accuracy of fixed SVC on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_fixed_train)*100, 2)) + ' %.')
    print('The accuracy of fixed SVC on the TEST is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_fixed_test)*100, 2)) + ' %.')
    dump_model(svc_fixed)

# Output for 1000 unigrams:
# The accuracy of default SVC on the TRAIN set is: 34.97 %.
# The accuracy of default SVC on the TEST is: 35.64 %.
# The accuracy of tuned SVC on the TRAIN set is: 73.64 %.
# The accuracy of tuned SVC on the TEST is: 72.33 %.

# Output for 300 unigrams:
# he accuracy of default SVC on the TRAIN set is: 56.01 %.
# The accuracy of default SVC on the TEST is: 57.65 %.
# The accuracy of tuned SVC on the TRAIN set is: 73.5 %.
# The accuracy of tuned SVC on the TEST is: 72.43 %.
start_time = time.process_time()
end_time = time.process_time() - start_time
