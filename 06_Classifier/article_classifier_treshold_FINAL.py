import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


def sort_probs(probs, cat):
    if 'article_number' in probs.columns:
        return probs.sort_values(by=cat, ascending=False)[[cat, 'topic_code', 'article_number']]
    else:
        return probs.sort_values(by=cat, ascending=False)[[cat, 'topic_code']]


def classify_articles(model, model_names, model_number, features_train, features_test, labels_train, labels_test,
                      article_numbers, stats, topx, threshold, dataset):
    # fitting/training the model on the proved training data
    model.fit(features_train, labels_train)
    probab_test = model.predict_proba(features_test)
    df_proba = pd.DataFrame(labels_test)
    predicted_classes_test = model.predict(features_test)
    f1s = f1_score(labels_test, predicted_classes_test, average=None)
    recalls = recall_score(labels_test, predicted_classes_test, average=None)
    precisions = precision_score(labels_test, predicted_classes_test, average=None)
    print('\nClassifier: ' + model_names[model_number] + ' accuracy: ' +
          str(round(accuracy_score(labels_test, predicted_classes_test)*100, 2)) + ' %.\n')
    print('\nClassifier: ' + model_names[model_number] + ' f1 macro: ' +
          str(round(f1_score(labels_test, predicted_classes_test, average='macro') * 100, 2)) + ' %.\n')
    print('\nClassifier: ' + model_names[model_number] + ' precision macro: ' +
          str(round(precision_score(labels_test, predicted_classes_test, average='macro') * 100, 2)) + ' %.\n')
    print('\nClassifier: ' + model_names[model_number] + ' recall macro: ' +
          str(round(recall_score(labels_test, predicted_classes_test, average='macro') * 100, 2)) + ' %.\n')
    print('Performance on ' + dataset + '.\n')

    for i in range(probab_test.shape[1]):
        column = "p" + str(i)
        df_proba[column] = probab_test[:, i]

    if article_numbers is not None:
        df_proba['article_number'] = article_numbers

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
            print('\nF1 : ' + str(f1s[i]) + ', precision: ' + str(precisions[i]) + ', recall: ' + str(recalls[i]) + '.')
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
path_data3 = path_project + '04_2_BinaryClassification/'

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
with open(path_data2 + 'df_test.pickle', 'rb') as data:
    df_test = pickle.load(data)
with open(path_data2 + 'df_train.pickle', 'rb') as data:
    df_train = pickle.load(data)
with open(path_data2 + 'df_test_unfiltered.pickle', 'rb') as data:
    df_test_unfiltered = pickle.load(data)
with open(path_data2 + 'df_train_unfiltered.pickle', 'rb') as data:
    df_train_unfiltered = pickle.load(data)
with open(path_data2 + 'tfidf_custom.pickle', 'rb') as data:
    tfidf_custom = pickle.load(data)
with open(path_data3 + 'df_test_predicted_relevant.pickle', 'rb') as data:
    df_test_predicted_relevant = pickle.load(data)
with open(path_data3 + 'df_val_predicted_relevant.pickle', 'rb') as data:
    df_val_predicted_relevant = pickle.load(data)
with open(path_data3 + 'df_val.pickle', 'rb') as data:
    df_val = pickle.load(data)

classifiers = [gbm, knn, mnb, rf, svm, nn]
classifiers_name = ['GradientBoost', 'NearestNeighbour', 'MultinomBayes', 'RandomForest', 'SupportVector',
                    'Multiperceptron']

# 1 == NN, 0 == SVM
modelselection = 1
show_filtered_results = 0
show_unfiltered_results = 0
# train a class unfiltered
with_cat_unfiltered = 0
show_final_results = 1
# show validation set
with_validation_set = 0
# show the stats for each categorie for true
show_stats = 1
# show the topX of each category
topX = 10
# make the threshold less agressive with a smaller number (range = [0,1])
threshold_multiplier = .9

if modelselection:
    classifiers = [nn]
    classifiers_name = ['Multiperceptron']
    # custom fit for NN
    if with_cat_unfiltered:
        thresholds = [0.83, 0.8, 0.9, 0.9, 0.987, 0.93, 0.98, 0.7, 0.9, 0.97, 0.9]
    else:
        thresholds = [0.83, 0.8, 0.9, 0.9, 0.987, 0.93, 0.98, 0.7, 0.9, 0.97]
    thresholds = [i * threshold_multiplier for i in thresholds]

else:
    classifiers = [svm]
    classifiers_name = ['SupportVector']
    # custom fit for SVM
    if with_cat_unfiltered:
        thresholds = [0.65, 0.7, 0.9, 0.9, 0.63, 0.8, 0.94, 0.9, 0.9, 0.97, 0.9]
    else:
        thresholds = [0.65, 0.7, 0.9, 0.9, 0.63, 0.8, 0.94, 0.9, 0.9, 0.97]
    thresholds = [i * threshold_multiplier for i in thresholds]

if show_filtered_results:
    # FILTERED DATA
    # FINAL test data set w/o irrelevant articles trained with FINAL training set w/o the irrelevant articles

    # fitting the tfidf transform on the whole train data
    features_train_filtered = tfidf_custom.fit_transform(df_train['article_words']).toarray()
    # applying the tfidf transform
    features_test_filtered = tfidf_custom.transform(df_test['article_words']).toarray()
    labels_train_filtered = df_train['topic_code']
    labels_test_filtered = df_test['topic_code']
    articles_num = df_test['article_number']

    num = 0
    for classifier in classifiers:
        classify_articles(classifier, classifiers_name, num, features_train_filtered, features_test_filtered,
                          labels_train_filtered, labels_test_filtered, articles_num, show_stats, topX, thresholds,
                          'FINAL TEST data set WITHOUT irrelevant articles')
        num += 1


if show_unfiltered_results:
    # UNFILTERED DATA
    # FINAL test data set WITH irrelevant articles trained with FINAL training set w/o the irrelevant articles

    if with_cat_unfiltered:
        # fitting the tfidf transform on the whole train data
        features_train_filtered = tfidf_custom.fit_transform(df_train_unfiltered['article_words']).toarray()
        labels_train_filtered = df_train_unfiltered['topic_code']
    else:
        # fitting the tfidf transform on the whole train data
        features_train_filtered = tfidf_custom.fit_transform(df_train['article_words']).toarray()
        labels_train_filtered = df_train['topic_code']

    if with_validation_set:
        # applying the tfidf transform
        features_test_unfiltered = tfidf_custom.transform(df_val['article_words']).toarray()
        labels_test_unfiltered = df_val['topic_code']
        articles_num = df_val['article_number']
    else:
        # applying the tfidf transform
        features_test_unfiltered = tfidf_custom.transform(df_test_unfiltered['article_words']).toarray()
        labels_test_unfiltered = df_test_unfiltered['topic_code']
        articles_num = df_test_unfiltered['article_number']

    num = 0
    for classifier in classifiers:
        classify_articles(classifier, classifiers_name, num, features_train_filtered, features_test_unfiltered,
                          labels_train_filtered, labels_test_unfiltered, articles_num, show_stats, topX, thresholds,
                          'FINAL TEST data set WITH irrelevant articles')
        num += 1

if show_final_results:
    # Final TEST DATA from Binary classification
    # fitting the tfidf transform on the whole train data
    features_train_filtered = tfidf_custom.fit_transform(df_train['article_words']).toarray()
    labels_train_filtered = df_train['topic_code']
    # applying the tfidf transform
    if with_validation_set:
        features_test_filtered = tfidf_custom.transform(df_val_predicted_relevant['article_words']).toarray()
        labels_test_filtered = df_val_predicted_relevant['topic_code']
        articles_num = df_val_predicted_relevant['article_number']
    else:
        features_test_filtered = tfidf_custom.transform(df_test_predicted_relevant['article_words']).toarray()
        labels_test_filtered = df_test_predicted_relevant['topic_code']
        articles_num = df_test_predicted_relevant['article_number']

    num = 0
    for classifier in classifiers:
        classify_articles(classifier, classifiers_name, num, features_train_filtered, features_test_filtered,
                          labels_train_filtered, labels_test_filtered, articles_num, show_stats, topX, thresholds,
                          'FINAL TEST data set with binary classification')
        num += 1

