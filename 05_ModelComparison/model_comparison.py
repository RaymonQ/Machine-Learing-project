import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def create_table(models, model_names):
    table = pd.DataFrame()
    table['Name'] = model_names
    training_accuracies = []
    test_accuracies = []
    f1_weigthed = []
    f1_macro = []
    for model in models:
        pred_train = model.predict(features_train)
        pred_test = model.predict(features_test)
        accuracy_train = round(accuracy_score(labels_train, pred_train)*100, 2)
        accuracy_test = round(accuracy_score(labels_test, pred_test)*100, 2)
        f1_weighted_test = round(f1_score(labels_test, pred_test, average='weighted') * 100, 2)
        f1_macro_test = round(f1_score(labels_test, pred_test, average='macro') * 100, 2)
        f1_weigthed.append(f1_weighted_test)
        f1_macro.append(f1_macro_test)
        training_accuracies.append(accuracy_train)
        test_accuracies.append(accuracy_test)
    table['F1_macro Test'] = f1_macro
    table['Accuracy Training'] = training_accuracies
    table['Accuracy Test'] = test_accuracies

    return table


# specify your directory where the project is here
# path Yannick:
path_project = "/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/"

# importing objects from folder data in feature engineering
path_data = path_project + '04_ModelTraining/Models/'
path_data2 = path_project + '03_FeatureEngineering/Data/'

with open(path_data + 'GBM.pickle', 'rb') as data:
    gbm = pickle.load(data)
with open(path_data + 'KNN.pickle', 'rb') as data:
    knn = pickle.load(data)
with open(path_data + 'MNB.pickle', 'rb') as data:
    mnb = pickle.load(data)
with open(path_data + 'RF.pickle', 'rb') as data:
    rf = pickle.load(data)
with open(path_data + 'NN.pickle', 'rb') as data:
    nn = pickle.load(data)
with open(path_data + 'SVM.pickle', 'rb') as data:
    svm = pickle.load(data)
with open(path_data2 + 'labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)
with open(path_data2 + 'labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)
with open(path_data2 + 'features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)
with open(path_data2 + 'features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)

classifiers = [gbm, knn, mnb, rf, svm, nn]
classifiers_name = ['GradientBoost', 'NearestNeighbour', 'MultinomBayes', 'RandomForest', 'SupportVector',
                    'Multiperceptron']

# classifiers = [svm, nn]
# classifiers_name = ['SupportVector', 'Multiperceptron']

df_table = create_table(classifiers, classifiers_name)
df_table_sorted = df_table.sort_values(by='F1_macro Test', ascending=False)
print(df_table_sorted)

# saving the table
with open('df_table_sorted.pickle', 'wb') as output:
    pickle.dump(df_table_sorted, output)

# Output for 2000 features:
#                Name  Accuracy Training  Accuracy Test
# 5   Multiperceptron              94.96          77.15
# 0     GradientBoost              98.64          77.04
# 1  NearestNeighbour              82.79          76.73
# 4     SupportVector              90.63          76.52
# 3      RandomForest              98.58          74.32
# 2     MultinomBayes              79.30          72.43

#  Output for 1000 features
#                Name  Accuracy Training  Accuracy Test
# 5   Multiperceptron              92.97          76.83
# 4     SupportVector              88.75          76.21
# 1  NearestNeighbour              82.76          76.10
# 0     GradientBoost              98.58          75.16
# 3      RandomForest              98.56          74.63
# 2     MultinomBayes              78.83          74.00

# Output for 1000 features
#                Name  Accuracy Training  Accuracy Test
# 0     GradientBoost              98.58          76.31
# 4     SupportVector              82.48          76.10
# 5   Multiperceptron              92.81          76.00
# 1  NearestNeighbour              80.19          75.47
# 3      RandomForest              98.56          74.63
# 2     MultinomBayes              78.83          74.00

# Output for 1000 features with balancing classes
#
#                Name  Accuracy Training  Accuracy Test
# 0     GradientBoost              99.85          75.89
# 5   Multiperceptron              99.55          74.95
# 4     SupportVector              96.75          74.84
# 3      RandomForest              99.85          74.63
# 2     MultinomBayes              90.22          74.21
# 1  NearestNeighbour              90.00          64.15

#                Name  Accuracy Training  Accuracy Test  F1_weighted Test  \
# 4     SupportVector              82.49           75.8             75.93
# 5   Multiperceptron              93.11           75.0             75.14
# 1  NearestNeighbour              62.17           57.8             56.35
# 2     MultinomBayes              56.80           54.8             57.21
# 0     GradientBoost              53.54           53.2             52.31
# 3      RandomForest              53.52           52.8             49.82
#
#    F1_macro Test
# 4          74.91
# 5          74.80
# 1          39.28
# 2          38.86
# 0          32.10
# 3          31.85
