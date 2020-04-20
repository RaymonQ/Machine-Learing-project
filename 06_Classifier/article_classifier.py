import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


def sort_probs(probs, cat):
    return probs.sort_values(by=cat, ascending=False)[[cat, 'topic_code']]


def classify_articles(model, features, labels, stats, dataset):
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
        top10 = sort_probs(df_proba, column).head(10)
        hits = top10['topic_code'] == i
        overall_hits += hits.astype(int).sum()
        if stats:
            print(top10)
            print('\nCategory ' + inv_code_categories[i] + ': ' + str(hits.astype(int).sum()) + '/10.\n\n')

    print('OVERALL RESULT: ' + str(overall_hits) + '/' + str(int(probab_test.shape[1])*10) + ".")
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

classifiers = [gbm, knn, mnb, rf, svm]
classifiers_name = ['GradientBoost', 'NearestNeighbour', 'MultinomBayes', 'RandomForest', 'SupportVector']

# show the stats for each categorie for true
show_stats = 0

# get the performance for the test data set from the training data set w/o irrelevant articles
for classifier in classifiers:
    classify_articles(classifier, features_test, labels_test, show_stats,
                      'TEST data SET from TRAINING data WITHOUT irrelevant articles')


# get the performance for the FINAL test data set w/o the irrelevant articles

# applying the tfidf transform
features_final_test = tfidf_custom.transform(df_test['article_words']).toarray()
labels_final_test = df_test['topic_code']

for classifier in classifiers:
    classify_articles(classifier, features_final_test, labels_final_test, show_stats,
                      'FINAL TEST data set WITHOUT irrelevant articles')


# get the performance for the test data set from the training data set with the irrelevant articles

# split the unfiltered test set again (USE SAME test_size AS IN feature_engineering!)
# NOTE: of course the unfiltered test set and the filtered test set are not 100 percent matching each other since it
# cannot be assumed that the irrelevant articles are randomly distributed over the given Training set!
_, words_test_unfiltered, _, labels_test_unfiltered = train_test_split(df_train_unfiltered['article_words'],
                                                                       df_train_unfiltered['topic_code'],
                                                                       test_size=0.2, random_state=0)

# applying the tfidf transform
features_test_unfiltered = tfidf_custom.transform(words_test_unfiltered).toarray()

for classifier in classifiers:
    classify_articles(classifier, features_test_unfiltered, labels_test_unfiltered, 0,
                      'TEST data set from TRAINING data WITH irrelevant articles')


# get the performance for the FINAL test data set with the irrelevant articles
features_final_test_unfilterd = tfidf_custom.transform(df_test_unfiltered['article_words']).toarray()
labels_final_test_unfilterd = df_test_unfiltered['topic_code']

for classifier in classifiers:
    classify_articles(classifier, features_final_test_unfilterd, labels_final_test_unfilterd, 0,
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
