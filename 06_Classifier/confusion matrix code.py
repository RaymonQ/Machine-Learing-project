import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pickle


# specify your directory where the project is here
# path Tal:
# path_project = "/Users/TalWe/.vscode/COMP9417 Group Assignment/COMP9417-Group-Assignment/"
path_project = "/Users/yannickschnider/PycharmProjects/COMP9417-Group-Assignment/"


# specify the data you want to create confusion matrix for (x_test and y_test)
X_path = path_project + '03_FeatureEngineering/Data/final_features_test.pickle'

y_path = path_project + '03_FeatureEngineering/Data/final_labels_test.pickle'

with open(X_path, 'rb') as data:
    X_input = pickle.load(data)
with open(y_path, 'rb') as data:
    y_target = pickle.load(data)


# import desired classifier (used NN.pickle as example)
clf_path = path_project + '04_ModelTraining/Models/NN.pickle'

with open(clf_path, 'rb') as data:
    classifier = pickle.load(data)


# Set parameters for plot_confusion_matrix function

# Class Names
class_names = ["ARTS CULTURE ENTERTAINMENT","BIOGRAPHIES PERSONALITIES PEOPLE",
                "DEFENCE", "DOMESTIC MARKETS","FOREX MARKETS","HEALTH","MONEY MARKETS",
                "SCIENCE AND TECHNOLOGY","SHARE LISTINGS","SPORTS","IRRELEVANT"]

# Set Title
title = "Confusion Matrix for NN Classifier"

cfm_disp = plot_confusion_matrix(classifier, X_input, y_target,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true', xticks_rotation=90)

cfm_disp.ax_.set_title(title)

print(title)
print(cfm_disp.confusion_matrix)

plt.show()
