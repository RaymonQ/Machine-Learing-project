import pickle
import pandas as pd


def sort_probs(probs, cat):
    return probs.sort_values(by=cat, ascending=False)[[cat, 'topic_code']]


def classify_articles(model, features, labels, stats, topx, threshold, dataset):
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
    overall_hits_sug = 0
    overall_suggestions = 0
    for i in range(probab_test.shape[1]):
        column = "p" + str(i)
        top_x = sort_probs(df_proba, column).head(topx)
        suggestions_index = top_x[column] > threshold[i]
        suggestions = top_x[suggestions_index]
        overall_suggestions += suggestions.shape[0]
        hits = top_x['topic_code'] == i
        overall_hits += hits.astype(int).sum()
        hits_sug = suggestions['topic_code'] == i
        overall_hits_sug += hits_sug.astype(int).sum()
        if stats:
            print(top_x)
            print('\nCategory ' + inv_code_categories[i] + ': ' + str(hits.astype(int).sum()) + '/' + str(topx) +
                  '.\n\n')
            print(suggestions)
            print('\nCategory ' + inv_code_categories[i] + ' Suggestions: ' + str(hits_sug.astype(int).sum()) + '/' +
                  str(suggestions.shape[0]) + '.\n\n')

    print('OVERALL RESULT: ' + str(overall_hits) + '/' + str(int(probab_test.shape[1])*topx) + ".")
    print('OVERALL SUGGESTIONS : ' + str(overall_hits_sug) + '/' + str(overall_suggestions) + ".")
    if stats:
        print('(Performance on ' + dataset + '.)\n')

    return df_proba


# specify your directory where the project is here
# path Yannick:
path_project = "/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/"

# importing objects from folder data in feature engineering
path_data = path_project + '04_ModelTraining/Models/'
path_data2 = path_project + '03_FeatureEngineering/Data/'
path_data3 = path_project + 'Raymon/'

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
with open(path_data3 + 'df_test_predicted_relevant.pickle', 'rb') as data:
    df_test_predicted_relevant = pickle.load(data)


# classifiers = [gbm, knn, mnb, rf, svm, nn]
# classifiers_name = ['GradientBoost', 'NearestNeighbour', 'MultinomBayes', 'RandomForest', 'SupportVector',
#                     'Multiperceptron']

# 1 == NN, 0 == SVM
modelselection = 1
show_filtered_results = 0
show_unfiltered_results = 0
show_final_results = 1
# show the stats for each categorie for true
show_stats = 1
# show the topX of each category
topX = 10
# make the threshold less agressive with a smaller number (range = [0,1])
threshold_multiplier = .8

if modelselection:
    classifiers = [nn]
    classifiers_name = ['Multiperceptron']
    # custom fit for NN
    thresholds = [0.83, 0.8, 0.9, 0.9, 0.987, 0.93, 0.98, 0.7, 0.9, 0.97]
    thresholds = [i * threshold_multiplier for i in thresholds]
else:
    classifiers = [svm]
    classifiers_name = ['SupportVector']
    # custom fit for SVM
    thresholds = [0.65, 0.7, 0.9, 0.9, 0.63, 0.8, 0.94, 0.9, 0.9, 0.97]
    thresholds = [i * threshold_multiplier for i in thresholds]

if show_filtered_results:
    # get the performance for the test data set from the training data set w/o irrelevant articles
    for classifier in classifiers:
        classify_articles(classifier, features_test, labels_test, show_stats, topX, thresholds,
                          'TEST data SET from TRAINING data WITHOUT irrelevant articles')

    # get the performance for the FINAL test data set w/o the irrelevant articles

    # applying the tfidf transform
    features_final_test = tfidf_custom.transform(df_test['article_words']).toarray()
    labels_final_test = df_test['topic_code']

    for classifier in classifiers:
        classify_articles(classifier, features_final_test, labels_final_test, show_stats, topX, thresholds,
                          'FINAL TEST data set WITHOUT irrelevant articles')

if show_unfiltered_results:

    # get the performance for the FINAL test data set with the irrelevant articles
    features_final_test_unfiltered = tfidf_custom.transform(df_test_unfiltered['article_words']).toarray()
    labels_final_test_unfiltered = df_test_unfiltered['topic_code']

    for classifier in classifiers:
        classify_articles(classifier, features_final_test_unfiltered, labels_final_test_unfiltered, show_stats, topX,
                          thresholds, 'FINAL TEST data set WITH irrelevant articles')

if show_final_results:

    # get the performance for the FINAL test data set with the binary classification before
    features_test_filtered = tfidf_custom.transform(df_test_predicted_relevant['article_words']).toarray()
    labels_test_filtered = df_test_predicted_relevant['topic_code']

    for classifier in classifiers:
        classify_articles(classifier, features_test_filtered, labels_test_filtered, show_stats, topX,
                          thresholds, 'FINAL TEST data set binary classification')
