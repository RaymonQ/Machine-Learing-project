import pickle

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

# printing the first 3 samples to see if the import worked
print(df_train.head(3))
print(df_test.head(3))

# ADD CODE TO ANALYSE THE DATA HERE:

