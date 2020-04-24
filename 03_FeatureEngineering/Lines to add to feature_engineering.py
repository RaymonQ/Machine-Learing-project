# Lines to add to feature_engineering.py to create X_input and y_labels for FINAL test set data
final_features_test = tfidf_custom.transform(df_test_unfiltered['article_words']).toarray()
final_labels_test = df_test_unfiltered['topic_code']


# final_features_test
with open(path_project + '03_FeatureEngineering/Data/final_features_test.pickle', 'wb') as output:
    pickle.dump(final_features_test, output)

# final_labels_test
with open(path_project + '03_FeatureEngineering/Data/final_labels_test.pickle', 'wb') as output:
    pickle.dump(final_labels_test, output)