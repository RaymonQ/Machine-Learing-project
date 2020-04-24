import pandas as pd
import pickle


# specify your directory where the project is here
# path Yannick:
path_project = "/Users/TalWe/.vscode/COMP9417 Group Assignment/COMP9417-Group-Assignment/"
#"/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/"

# extended paths to the .csv files
path_trainingData = path_project + '00_TaskHandout/training.csv'
path_testData = path_project + '00_TaskHandout/test.csv'

# import the files as panda data frames
df_train = pd.read_csv(path_trainingData, sep=',')
df_test = pd.read_csv(path_testData, sep=',')

# print the first 5 entries of each to check on the imported data
print(df_train.head(5))
print(df_test.head(5))


# save the frames as pandas data frames using pickle so they can later be easily used by other python-programmes
# (open creates the files, if they don't exist yet)
with open('DataTrain.pickle', 'wb') as output:
    pickle.dump(df_train, output)

with open('DataTest.pickle', 'wb') as output:
    pickle.dump(df_test, output)

