import pickle
from sklearn.feature_extraction.text import TfidfVectorizer as TfIdf
from sklearn.model_selection import train_test_split

# specify your directory where the project is here
# path Yannick:
path_project = "/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/"

# importing the data frame using pickle
path_train = path_project + '01_ImportData/DataTrain.pickle'
path_test = path_project + '01_ImportData/DataTest.pickle'

with open(path_train, 'rb') as data:
    df_train = pickle.load(data)
with open(path_test, 'rb') as data:
    df_test = pickle.load(data)

# # printing the first 3 samples to see if the import worked
# print(df_train.head(3))
# print(df_test.head(3))

# ADD CODE FEATURE ENGINEERING HERE:
# structure of dataframe rows: 0,1...N columns: 'article_number', 'aricticle_words', 'topic'

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
df_train['topic_code'] = df_train['topic']
df_train = df_train.replace({'topic_code': codes_categories})

df_test['topic_code'] = df_test['topic']
df_test = df_test.replace({'topic_code': codes_categories})


# splitting our training data into a training and a test set to be able to validate our methods without
# using the real test set...

# here 80 percent belong to the training set and 20 to the new test set
words_train, words_test, labels_train, labels_test = train_test_split(df_train['article_words'], df_train['topic_code'],
                                                                      test_size=0.2, random_state=0)
# # check on the sizes of the sets:
# print(words_train.shape)
# print(labels_train.shape)
# print(words_test.shape)
# print(labels_test.shape)


# converting the words to TF-IDF vectors

tfidf_default = TfIdf()
# # evaluate default parameters
# print(tfidf_default.get_params())

# parameters (to be tuned)

# default: (1,1) -> only unigrams , (1,2) -> uni and bigrams, (2,2) -> only bigrams...
ngram_range = (1, 1)
# default: 1 (range[0,1]) -> ignore terms that have a lower document frequency
min_df = 1
# default: 1 (range[0,1]) -> ignore terms that have a higher document frequency
max_df = 1
# default = None -> max number of words to be taken into account
# setting it to 500 to prevent to large vector-pickles files
max_features = 400
# default = False -> apply sublinear transfer function
sublinear_tf = False

# constructing custom TFIDF object with the above parameters
tfidf_custom = TfIdf(ngram_range=ngram_range, max_df=max_df, min_df=min_df, max_features=max_features,
                     sublinear_tf=sublinear_tf)
# FITTING the TFDIF vectorizer on our training data AND APPLYING the transform
features_train = tfidf_custom.fit_transform(words_train).toarray()
# only APPLYING the transform on our test data (we want the same features as for the training data!)
features_test = tfidf_custom.transform(words_test).toarray()

# checking on the dimensions of the arrays
print(features_train.shape)
print(features_test.shape)

# Finally we export and save our train and test data with pickle in the Folder Data

# df_train
with open('Data/df_train.pickle', 'wb') as output:
    pickle.dump(df_train, output)

# df_test
with open('Data/df_test.pickle', 'wb') as output:
    pickle.dump(df_test, output)

# features_train
with open('Data/features_train.pickle', 'wb') as output:
    pickle.dump(features_train, output)

# labels_train
with open('Data/labels_train.pickle', 'wb') as output:
    pickle.dump(labels_train, output)

# features_test
with open('Data/features_test.pickle', 'wb') as output:
    pickle.dump(features_test, output)

# labels_test
with open('Data/labels_test.pickle', 'wb') as output:
    pickle.dump(labels_test, output)

# tfidf object
with open('Data/tfidf_custom.pickle', 'wb') as output:
    pickle.dump(tfidf_custom, output)
