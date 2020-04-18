import pickle
import pandas as pd


def sort_probs(probs, cat):
    return probs.sort_values(by=cat, ascending=False)[[cat, 'topic_code']]


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

classifiers = [gbm, knn, mnb, rf, svm]
classifiers_name = ['GradientBoost', 'NearestNeighbour', 'MultinomBayes', 'RandomForest', 'SupportVector']

probab_test = svm.predict_proba(features_test)
df_proba = pd.DataFrame(labels_test)

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
    print(top10)
    print('Category ' + inv_code_categories[i] + ': ' + str(hits.astype(int).sum()) + '/10.')

print('\nOverall result: ' + str(overall_hits) + '/100.')
