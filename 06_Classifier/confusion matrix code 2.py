# Confusion Matrix 2
import pickle

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

# specify your directory where the project is here
# path Tal:
path_project = "/Users/TalWe/.vscode/COMP9417 Group Assignment/COMP9417-Group-Assignment/"

pred_path = "specify path of pickle file containing predictions"
actual_path = "specify path of pickle file containing actual classes of those predictions"

with open(path_project + pred_path, 'rb') as data:
    y_pred = pickle.load(data)
with open(path_project + actual_path, 'rb') as data:
    y_actual = pickle.load(data)

cnf_matrix = confusion_matrix(y_true=y_actual, y_pred=y_pred)

class_names = ["ARTS CULTURE ENTERTAINMENT","BIOGRAPHIES PERSONALITIES PEOPLE",
                "DEFENCE", "DOMESTIC MARKETS","FOREX MARKETS","HEALTH","MONEY MARKETS",
                "SCIENCE AND TECHNOLOGY","SHARE LISTINGS","SPORTS","IRRELEVANT"]

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14, normalize=False):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    if normalize:
        confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fmt = '.2f' if normalize else 'd'

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt=fmt)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

print_confusion_matrix(confusion_matrix=cnf_matrix,class_names=class_names,normalize=True)