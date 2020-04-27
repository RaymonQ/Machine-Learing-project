import pickle

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.linear_model import LogisticRegression
import time

from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# Decide if to implement grid search or not
fit_grid_search = 0
tfidf_cv = 0

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
max_features = 2000
sublinear_tf = True
stop_words = None
tfidf_custom = TfIdf(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features,
                     sublinear_tf=sublinear_tf, stop_words = stop_words)

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

# here valdiation set split
df_train, df_val = train_test_split(df, test_size=500, random_state=0)
Xtr = tfidf_custom.fit_transform(df_train['article_words']).toarray()
Xval = tfidf_custom.transform(df_val['article_words']).toarray()
Ytr = df_train['relevance']
Yval = df_val['relevance']

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
    max_iter = [100,200,300,500]

    param_grid = [
        {'penalty': ['l2'], 'multi_class': multiclass, 'solver': ['newton-cg','sag','lbfgs'], 'n_jobs': n_jobs, 'max_iter': max_iter},
        {'penalty': ['elasticnet'], 'multi_class': multiclass, 'solver': ['saga'], 'n_jobs': n_jobs, 'max_iter': max_iter},
        {'penalty': ['l1'], 'multi_class': multiclass, 'solver': ['liblinear','saga'], 'n_jobs': n_jobs, 'max_iter': max_iter}
    ]

    # Create a base model
    logit_model = LogisticRegression()

    # splits in CV with random state
    cv_sets = ShuffleSplit(n_splits=5, test_size=.2, random_state=0)

    # Instantiate the grid search model
    start_time = time.process_time()
    grid_search = GridSearchCV(estimator=logit_model,
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


# Reference:
# https://stackoverflow.com/questions/44066264/how-to-choose-parameters-in-tfidfvectorizer-in-sklearn-during-unsupervised-clust
# Utilised method of gridsearchcv to tune TF-IDF hyperparameters
# User - David Batista
if tfidf_cv:
    pipeline = Pipeline([
        ('tfidf', TfIdf(stop_words=None,sublinear_tf=True,max_df=1.,ngram_range=(1,1))),
        ('clf', LogisticRegression())
    ])
    
    parameters = {
        'tfidf__min_df': (1,5,10,20),
        'tfidf__max_features': (100,500,1000,2000,5000,10000)
    }
    
    scoring = {'AUC': 'roc_auc', 'F1': 'f1', 'Precision': 'precision', 'Recall': 'recall'}
    
    start_time = time.process_time()
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1,verbose=1, scoring=scoring, refit='F1')
    grid_search_tune.fit(df['article_words'],labels_train)
    end_time = time.process_time() - start_time

# Real test data
# parameters chosen from Cross Validation
logit_model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1,
          penalty='l2', random_state=None, solver='sag', tol=0.0001,
          verbose=0, warm_start=False)

# Fit model 
logit_model = logit_model.fit(features_train, labels_train)
prediction = logit_model.predict(X_test_final)

logit_model_val = logit_model.fit(Xtr,Ytr)
pred_val = logit_model.predict(Xval)

# Confusion metrics
cnf_matrix = confusion_matrix(y_test_final, prediction)
cnf_matrix_val = confusion_matrix(Yval, pred_val)
print(cnf_matrix)
print(cnf_matrix_val)

# Results
print(classification_report(y_test_final, prediction))
print(classification_report(Yval, pred_val))

# Filter Test Data to only include articles that were prediction relevant by logistic model function
df_test['prediction'] = prediction
df_test_predicted_relevant = df_test[df_test['prediction'] == 1]

# Filter validation set Data to only include articles that were prediction relevant by logistic model function
df_val['prediction'] = pred_val
df_val_predicted_relevant = df_val[df_val['prediction'] == 1]

# Remove relevance and prediction columns (as they're redundant for next part of classification)
df_test_predicted_relevant = df_test_predicted_relevant.drop(columns=["relevance", "prediction"])
df_val_predicted_relevant = df_val_predicted_relevant.drop(columns=["relevance", "prediction"])
df_val = df_val.drop(columns=["relevance", "prediction"])


print(df_test_predicted_relevant.head())
print(df_val_predicted_relevant.head())
print(df_val.head())

# creating a ditionary with the labels
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

# mapping and creating a new column 'topic_code'
df_test_predicted_relevant['topic_code'] = df_test_predicted_relevant['topic']
df_test_predicted_relevant = df_test_predicted_relevant.replace({'topic_code': codes_categories})

df_val_predicted_relevant['topic_code'] = df_val_predicted_relevant['topic']
df_val_predicted_relevant = df_val_predicted_relevant.replace({'topic_code': codes_categories})

df_val['topic_code'] = df_val['topic']
df_val = df_val.replace({'topic_code': codes_categories})

print(df_test_predicted_relevant.head())
print(df_val_predicted_relevant.head())
print(df_val.head())
print(df_val.shape)

# Save Logistic Regression model in Pickle file
with open('df_test_predicted_relevant.pickle', 'wb') as output:
    pickle.dump(df_test_predicted_relevant, output)
with open('df_val_predicted_relevant.pickle', 'wb') as output:
    pickle.dump(df_val_predicted_relevant, output)
with open('df_val.pickle', 'wb') as output:
    pickle.dump(df_val, output)

    
### Creation of Graphs for Report ###

# Reference:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html
# Plotting Multiple metrics for Kfold (k=5) cross validation on a feature

### Plotting Multiple Metrics at same time for Max Features
results = grid_search_tune.cv_results_
scoring = {'AUC': 'roc_auc', 'F1': 'f1', 'Precision': 'precision', 'Recall': 'recall'}

plt.figure(figsize=(13, 13))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("Max features")
plt.ylabel("Score")

ax = plt.gca()
ax.set_xlim(0, 10000)
ax.set_ylim(0.73, 1)

# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_tfidf__max_features'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k', 'y','r']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
    ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
plt.grid(False)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
min_df = (1,5,10,20)
max_features = (100,500,1000,2000,5000,10000)


# Reference for plot_grid_search:
# https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
# User - Mike Lewis
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average ROC_AUC Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
# Calling Method 
plot_grid_search(grid_search_tune.cv_results_, max_features, min_df, 'Max Features', 'min_df')
