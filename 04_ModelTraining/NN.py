import pickle
from sklearn.neural_network import MLPClassifier as Mlp
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ShuffleSplit
import time


def dump_model(model):
    with open('Models/NN.pickle', 'wb') as output:
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

calc_default_model = 0
calc_tuned_model = 0
calc_fixed_model = 1

if calc_default_model:
    # see default parameters of RF classifier
    mlp_default = Mlp(random_state=0)
    # hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.
    print(mlp_default.get_params())

    # fit the model
    mlp_default.fit(features_train, labels_train)

    predicted_classes_train = mlp_default.predict(features_train)
    predicted_classes_test = mlp_default.predict(features_test)

    print('The accuracy of the default MLP classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')

    print('The accuracy of the default MLP classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
    dump_model(mlp_default)

if calc_tuned_model:
    mlp_model = Mlp(max_iter=500, random_state=0)

    param_grid = {
        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant', 'adaptive'],
    }
    grid_search = GridSearchCV(mlp_model, param_grid, n_jobs=-1, cv=3, verbose=10)
    grid_search.fit(features_train, labels_train)

    mlp_best_grid = grid_search.best_estimator_
    predicted_classes_train = mlp_best_grid.predict(features_train)
    predicted_classes_test = mlp_best_grid.predict(features_test)

    # Best paramete set
    print('Best parameters found:\n', grid_search.best_params_)
    print("\nAccuracy with best parameters:")
    print(grid_search.best_score_)

    print('The accuracy of the best Mlp classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')
    print('The accuracy of the best Mlp classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
    dump_model(mlp_best_grid)

    # Best parameters found:
    #  {'activation': 'tanh', 'alpha': 0.05, 'hidden_layer_sizes': (100,), 'learning_rate': 'constant',
    # 'solver': 'adam'}
    # Accuracy with best parameters:
    # 0.74501573976915
    # The accuracy of the best Mlp classifier on the TRAIN set is: 86.41 %.
    # The accuracy of the best Mlp classifier on the TEST set is: 74.21 %.

if calc_fixed_model:
    # see default parameters of RF classifier
    mlp_fixed = Mlp(hidden_layer_sizes=(300,), random_state=0, max_iter=500, solver='adam', learning_rate='constant',
                    activation='tanh', alpha=.05)
    # hidden_layer_sizes: The ith element represents the number of neurons in the ith hidden layer.

    # fit the model
    mlp_fixed.fit(features_train, labels_train)

    predicted_classes_train = mlp_fixed.predict(features_train)
    predicted_classes_test = mlp_fixed.predict(features_test)

    print('The accuracy of the fixed MLP classifier on the TRAIN set is: ' +
          str(round(accuracy_score(labels_train, predicted_classes_train) * 100, 2)) + ' %.')

    print('The accuracy of the fixed MLP classifier on the TEST set is: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test) * 100, 2)) + ' %.')
    dump_model(mlp_fixed)

# unigrams:
# The accuracy of the fixed MLP classifier on the TRAIN set is: 92.81 %.
# The accuracy of the fixed MLP classifier on the TEST set is: 76.0 %.
# with bigrams:
# The accuracy of the fixed MLP classifier on the TRAIN set is: 92.44 %.
# The accuracy of the fixed MLP classifier on the TEST set is: 74.63 %.

# unigrams, minority:
# The accuracy of the fixed MLP classifier on the TRAIN set is: 96.05 %.
# The accuracy of the fixed MLP classifier on the TEST set is: 76.31 %.
