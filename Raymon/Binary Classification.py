import pandas as pd
# read data
path_project = "/Users/chengqian/Desktop/COMP9417-Group-Assignment/"
path_trainingData = path_project + '00_TaskHandout/training.csv'
df = pd.read_csv(path_trainingData, sep=',')

#TfIdf settings
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
ngram_range = (1, 1)
min_df = 10
max_df = 1.
max_features = 300
sublinear_tf = True
tfidf_custom = TfIdf(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features,
                     sublinear_tf=sublinear_tf)

#Create "relevance" label
def label_relevance (row):
   if row['topic'] == 'IRRELEVANT' :
      return '0'
   else:
      return '1'

#Add "relevance" label
df.apply (lambda row: label_relevance(row), axis=1)
df['relevance'] = df.apply (lambda row: label_relevance(row), axis=1)
X = tfidf_custom.fit_transform(df['article_words']).toarray()
y = df_train['relevance']

#Split traing and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LogisticRegression
logit_model = LogisticRegression()
# Fit model 
logit_model = logit_model.fit(X_train, y_train)

prediction = logit_model.predict(X_test)
# Confusion metrics
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, prediction)
print(confusion_matrix)
# Results
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))
