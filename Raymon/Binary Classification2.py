import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
import time
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier as Mlp
import os
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.naive_bayes import MultinomialNB


# Decide if to implement grid search or not
fit_grid_search = 0

# Specify Path Project
# path_project = "/Users/TalWe/.vscode/COMP9417 Group Assignment/COMP9417-Group-Assignment/"
path_project = "/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/"

# read data
df = pd.read_csv(path_project + "00_TaskHandout/training.csv", sep=',')
df_test = pd.read_csv(path_project + "00_TaskHandout/test.csv", sep=',')

# TfIdf settings
ngram_range = (1, 1)
min_df = 10
max_df = 1.
max_features = 400
sublinear_tf = True
stop_words = None
tfidf_custom = TfIdf(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features,
                     sublinear_tf=sublinear_tf, stop_words=stop_words)

topic_codes = {
    'IRRELEVANT': 0,
    'ARTS CULTURE ENTERTAINMENT': 1,
    'BIOGRAPHIES PERSONALITIES PEOPLE': 1,
    'DEFENCE': 1,
    'DOMESTIC MARKETS': 1,
    'FOREX MARKETS': 1,
    'HEALTH': 1,
    'MONEY MARKETS': 1,
    'SCIENCE AND TECHNOLOGY': 1,
    'SHARE LISTINGS': 1,
    'SPORTS': 1,   
}

df['relevance'] = df['topic']
df = df.replace({'relevance': topic_codes})

df_test['relevance'] = df_test['topic']
df_test = df_test.replace({'relevance': topic_codes})

# Fit TFIDF to Train Features, only apply transform to Test input
features_train = tfidf_custom.fit_transform(df['article_words']).toarray()
X_test_final = tfidf_custom.transform(df_test['article_words']).toarray()
labels_train = df['relevance']
y_test_final = df_test['relevance']


# IMPLEMENT CROSS VALIDATION FOR TUNING LOGISTIC REGRESSION HYPERPARAMETERS
if fit_grid_search:
    # parameter grid based on output random search
    multiclass = ['ovr']
    n_jobs = [-1]
    max_iter = [100, 200, 300, 500]

    param_grid = [
        {'penalty': ['l2'], 'multi_class': multiclass, 'solver': ['newton-cg', 'sag', 'lbfgs'], 'n_jobs': n_jobs,
         'max_iter': max_iter},
        {'penalty': ['elasticnet'], 'multi_class': multiclass, 'solver': ['saga'], 'n_jobs': n_jobs,
         'max_iter': max_iter},
        {'penalty': ['l1'], 'multi_class': multiclass, 'solver': ['liblinear', 'saga'], 'n_jobs': n_jobs,
         'max_iter': max_iter}
    ]

    # Create a base model
    model = LogisticRegression()

    # splits in CV with random state
    cv_sets = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

    # Instantiate the grid search model
    start_time = time.process_time()
    grid_search = GridSearchCV(estimator=model,
                               param_grid=param_grid,
                               scoring='accuracy',
                               cv=cv_sets,
                               verbose=0)

    # Fit the grid search to the data
    grid_search.fit(features_train, labels_train)
    end_time = time.process_time() - start_time

    logit_best_grid = grid_search.best_estimator_
    predicted_classes_train = logit_best_grid.predict(features_train)
    predicted_classes_test = logit_best_grid.predict(X_test_final)

# Real test data
# parameters chosen from Cross Validation

# model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                                  intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
#                                  penalty='l2', random_state=None, solver='sag', tol=0.0001,
#                                  verbose=0, warm_start=False)

# model = Mlp(hidden_layer_sizes=(100,100), random_state=1, max_iter=500, solver='adam', learning_rate='constant',
#                     activation='tanh', alpha=.05)

# model = MultinomialNB()
# model = Knn(n_neighbors=30)

# #use GridSearch to better choose tuning parameters C and gamma
# param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
# model = GridSearchCV(SVC(), param_grid, verbose=3,n_jobs=3,cv=2)
# model.fit(features_train,labels_train)
#
# print(model.best_params_)
# print(model.best_estimator_)
model = SVC(C=1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
            decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
            probability=False, random_state=None, shrinking=True, tol=0.001,
            verbose=False)

# Fit model 
model = model.fit(features_train, labels_train)
prediction = model.predict(X_test_final)

# Confusion metrics
cnf_matrix = confusion_matrix(y_test_final, prediction)
print(cnf_matrix)

# Results
print(classification_report(y_test_final, prediction))

# Filter Test Data to only include articles that were prediction relevant by logistic model function
df_test['prediction'] = prediction
df_test_predicted_relevant = df_test[df_test['prediction'] == 1]

# Remove relevance and prediction columns (as they're redundant for next part of classification)
df_test_predicted_relevant = df_test_predicted_relevant.drop(columns=["relevance", "prediction"])

# print(df_test_predicted_relevant.head())

os.system('say "your program has finished"')

# result logistic regression:
# [[238  28]
#  [ 28 206]]
#               precision    recall  f1-score   support
#
#            0       0.89      0.89      0.89       266
#            1       0.88      0.88      0.88       234
#
#     accuracy                           0.89       500
#    macro avg       0.89      0.89      0.89       500
# weighted avg       0.89      0.89      0.89       500
# # creating a ditionary with the labels
# codes_categories = {'ARTS CULTURE ENTERTAINMENT': 0,
#                     'BIOGRAPHIES PERSONALITIES PEOPLE': 1,
#                     'DEFENCE': 2,
#                     'DOMESTIC MARKETS': 3,
#                     'FOREX MARKETS': 4,
#                     'HEALTH': 5,
#                     'MONEY MARKETS': 6,
#                     'SCIENCE AND TECHNOLOGY': 7,
#                     'SHARE LISTINGS': 8,
#                     'SPORTS': 9,
#                     'IRRELEVANT': 10}
#
# # mapping and creating a new column 'topic_code'
# df_test_predicted_relevant['topic_code'] = df_test_predicted_relevant['topic']
# df_test_predicted_relevant = df_test_predicted_relevant.replace({'topic_code': codes_categories})
#
# print(df_test_predicted_relevant.head())

#
# # Save Logistic Regression model in Pickle file
# with open('df_test_predicted_relevant.pickle', 'wb') as output:
#     pickle.dump(df_test_predicted_relevant, output)
