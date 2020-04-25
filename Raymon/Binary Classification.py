import pandas as pd
import pickle
# read data
path_project = "/Users/chengqian/Desktop/COMP9417-Group-Assignment/"
path_trainingData = path_project + '00_TaskHandout/training.csv'
path_testData = path_project + '00_TaskHandout/test.csv'

df = pd.read_csv(path_trainingData, sep=',')
df_test = pd.read_csv(path_testData, sep=',')

#TfIdf settings
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
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
df = df.replace({'relevance':topic_codes})

df_test['relevance'] = df_test['topic']
df_test = df_test.replace({'relevance':topic_codes})

# Fit TFIDF to Train Features, only apply transform to Test input
features_train = tfidf_custom.fit_transform(df['article_words']).toarray()
X_test_final = tfidf_custom.transform(df_test['article_words']).toarray()
labels_train = df['relevance']
y_test_final = df_test['relevance']



# Train Validation Split on Training Data to apply Cross Validation to
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

for i in range(10):
    X_train,X_test,y_train,y_test=train_test_split(features_train,labels_train,test_size=0.20,random_state=None)

    # Validation test data
    logit_model = LogisticRegression()

    # Fit model 
    logit_model = logit_model.fit(X_train, y_train)
    prediction = logit_model.predict(X_test)
    
    # Confusion metrics
    cnf_matrix = confusion_matrix(y_test, prediction)
    print(cnf_matrix)
    
    # Results
    
    print(classification_report(y_test, prediction))


# Real test data
logit_model = LogisticRegression()

# Fit model 
logit_model = logit_model.fit(features_train, labels_train)
prediction = logit_model.predict(X_test_final)

# Confusion metrics
cnf_matrix = confusion_matrix(y_test_final, prediction)
print(cnf_matrix)

# Results
print(classification_report(y_test_final, prediction))


# Save Logistic Regression model in Pickle file
with open(path_project + 'Raymon/binaryModel.pickle', 'wb') as output:
    pickle.dump(logit_model, output)
