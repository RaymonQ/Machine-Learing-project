import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


def sort_probs(probs, cat):
    return probs.sort_values(by=cat, ascending=False)[[cat, 'topic_code']]


def classify_articles(model, features, labels, stats, topx, dataset):
    print('\nClassifier: ' + str(model) + '\n')
    print('Performance on ' + dataset + '.\n')
    probab_test = model.predict_proba(features)
    df_proba = pd.DataFrame(labels)

    for i in range(probab_test.shape[1]):
        column = "p" + str(i)
        df_proba[column] = probab_test[:, i]

    codes_categories = {'ARTS CULTURE ENTERTAINMENT': 0,
                        'BIOGRAPHIES PERSONALITIES PEOPLE': 1,
                        'DEFENCE': 2,
                        'DOMESTIC MARKETS': 3,
                        'FOREX MARKETS': 4,
                        'HEALTH': 5,
                        'MONEY MARKETS': 6,
                        'SCIENCE AND TECHNOLOGY': 7,
                        'SHARE LISTINGS': 8,
                        'SPORTS': 9,
                        'IRRELEVANT': 10}

    inv_code_categories = {v: k for k, v in codes_categories.items()}

    overall_hits = 0
    for i in range(probab_test.shape[1]):
        column = "p" + str(i)
        top_x = sort_probs(df_proba, column).head(topx)
        hits = top_x['topic_code'] == i
        overall_hits += hits.astype(int).sum()
        if stats:
            print(top_x)
            print('\nCategory ' + inv_code_categories[i] + ': ' + str(hits.astype(int).sum()) + '/' + str(topx) +
                  '.\n\n')

    print('OVERALL RESULT: ' + str(overall_hits) + '/' + str(int(probab_test.shape[1])*topx) + ".")
    if stats:
        print('(Performance on ' + dataset + '.)\n')

    return df_proba


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
with open(path_data + 'SVM.pickle', 'rb') as data:
    svm = pickle.load(data)
with open(path_data + 'NN.pickle', 'rb') as data:
    nn = pickle.load(data)
with open(path_data2 + 'labels_train.pickle', 'rb') as data:
    labels_train = pickle.load(data)
with open(path_data2 + 'labels_test.pickle', 'rb') as data:
    labels_test = pickle.load(data)
with open(path_data2 + 'features_train.pickle', 'rb') as data:
    features_train = pickle.load(data)
with open(path_data2 + 'features_test.pickle', 'rb') as data:
    features_test = pickle.load(data)
with open(path_data2 + 'df_test.pickle', 'rb') as data:
    df_test = pickle.load(data)
with open(path_data2 + 'df_test_unfiltered.pickle', 'rb') as data:
    df_test_unfiltered = pickle.load(data)
with open(path_data2 + 'df_train_unfiltered.pickle', 'rb') as data:
    df_train_unfiltered = pickle.load(data)
with open(path_data2 + 'tfidf_custom.pickle', 'rb') as data:
    tfidf_custom = pickle.load(data)

show_unfiltered_stats = 1

classifiers = [gbm, knn, mnb, rf, svm, nn]
classifiers = [svm, nn]
classifiers_name = ['GradientBoost', 'NearestNeighbour', 'MultinomBayes', 'RandomForest', 'SupportVector',
                    'Multiperceptron']

# show the stats for each categorie for true
show_stats = 0
# show the topX of each category
topX = 10

# get the performance for the test data set from the training data set w/o irrelevant articles
for classifier in classifiers:
    classify_articles(classifier, features_test, labels_test, show_stats, topX,
                      'TEST data SET from TRAINING data WITHOUT irrelevant articles')


# get the performance for the FINAL test data set w/o the irrelevant articles

# applying the tfidf transform
features_final_test = tfidf_custom.transform(df_test['article_words']).toarray()
labels_final_test = df_test['topic_code']

for classifier in classifiers:
    classify_articles(classifier, features_final_test, labels_final_test, show_stats, topX,
                      'FINAL TEST data set WITHOUT irrelevant articles')

if show_unfiltered_stats:

    # get the performance for the FINAL test data set with the irrelevant articles
    features_final_test_unfilterd = tfidf_custom.transform(df_test_unfiltered['article_words']).toarray()
    labels_final_test_unfilterd = df_test_unfiltered['topic_code']

    for classifier in classifiers:
        classify_articles(classifier, features_final_test_unfilterd, labels_final_test_unfilterd, show_stats, topX,
                          'FINAL TEST data set WITH irrelevant articles')

# Results on test set from training set trained WITH Categorie Irrelevant:
# GBM = 73/110
# Knn = 75/110
# MNB = 73/110
# RF =  71/110
# SVC = 84/110

# Results on FINAL test trained WITH Categorie Irrelevant:
# GBM = 52/110
# Knn = 56/110
# MNB = 54/110
# RF =  56/110
# SVC = 62/110


# Results on FINAL test trained WITHOUT Categorie Irrelevant:
# GBM = 47/100
# Knn = 42/100
# MNB = 45/100
# RF =  47/100
# SVC = 51/100

# Results 2000 features, on final test set:
# GBM = 67/100
# Knn = 65/100
# MNB = 68/100
# RF =  66/100
# SVC = 70/100
# MLP = 70/100

# Results 1000 features, on final test set:
# GBM = 70/100
# Knn = 66/100
# MNB = 67/100
# RF =  64/100
# SVC = 69/100
# MLP = 71/100

# Results 1000 features, on final test set with 95 quantile:
# GBM = 66/100
# Knn = 61/100
# MNB = 66/100
# RF =  64/100
# SVC = 69/100
# MLP = 69/100

# Results for 1000 unigrams Final test set:
# SVC = 67/100
# MLP = 70/100
# with unfiltered articles in Final set:
# SVC = 56/100
# MLP = 53/100
