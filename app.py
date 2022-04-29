from sklearn.linear_model import LogisticRegression
import streamlit as st 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from sklearn import metrics
from sklearn.metrics import accuracy_score

st.set_option('deprecation.showPyplotGlobalUse', False)
plt.rcParams["figure.figsize"] = (5,5)

st.title('Final Project : Distributed & Scalable Data Engineering')
st.title('Project Members : Deepak Singh')

df = pd.read_csv("heart.csv")

dataset_name = "Heart Disease Dataset"

st.write(f"## {dataset_name} Dataset")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression', 'Support Vector Machine', 'Random Forest','K Nearest Neighbors')
)

def get_dataset():
    y = df["condition"]
    df.drop(columns="condition",inplace=True)
    X = df
    return X, y

X, y = get_dataset()
st.write('Shape of Dataset:', X.shape)
st.write('Number of Classes:', len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'Support Vector Machine':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'K Nearest Neighbors':
        K = st.sidebar.slider('K', 1, 15)
        params['K'] = K
    elif clf_name == 'Logistic Regression':
        pass
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'Support Vector Machine':
        clf = SVC(C=params['C'])
    elif clf_name == 'K Nearest Neighbors':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'Logistic Regression':
        clf = LogisticRegression()
    else:
        clf = clf = RandomForestClassifier(n_estimators=params['n_estimators'], 
            max_depth=params['max_depth'], random_state=1234)
    return clf

clf = get_classifier(classifier_name, params)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
from sklearn.metrics import f1_score
fscore = f1_score(y_test,y_pred)

st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc)
st.write(f'F1 Score =',fscore)


target_names = ["Class 0","Class 1"]
x = classification_report(y_test,y_pred,target_names=target_names)

def show_values(pc, fmt="%.2f", **kw):
  
    pc.update_scalarmappable()
    ax = pc.axes
    for p, color, value in zip(pc.get_paths(), pc.get_facecolors(), pc.get_array()):
        x, y = p.vertices[:-2, :].mean(0)
        if np.all(color[:3] > 0.5):
            color = (0.0, 0.0, 0.0)
        else:
            color = (1.0, 1.0, 1.0)
        ax.text(x, y, fmt % value, ha="center", va="center", color=color, **kw)


def cm2inch(*tupl):
  
    inch = 2.54
    if type(tupl[0]) == tuple:
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def heatmap(AUC, title, xlabel, ylabel, xticklabels, yticklabels, figure_width=40, figure_height=20, correct_orientation=False, cmap='RdBu'):
  

    fig, ax = plt.subplots()    
    c = ax.pcolor(AUC, edgecolors='k', linestyle= 'dashed', linewidths=0.2, cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_yticks(np.arange(AUC.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(AUC.shape[1]) + 0.5, minor=False)

    ax.set_xticklabels(xticklabels, minor=False)
    ax.set_yticklabels(yticklabels, minor=False)

    plt.title(title, y=1.25)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)      

    plt.xlim( (0, AUC.shape[1]) )

    ax = plt.gca()    
    for t in ax.xaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)
    for t in ax.yaxis.get_major_ticks():
        t.tick1line.set_visible(False)
        t.tick2line.set_visible(False)

    plt.colorbar(c)

    show_values(c)

    if correct_orientation:
        ax.invert_yaxis()
        ax.xaxis.tick_top()       

    fig = plt.gcf()
    fig.set_size_inches(cm2inch(figure_width, figure_height))

def plot_classification_report(classification_report, number_of_classes=2, title='Classification report ', cmap='RdYlGn'):

    lines = classification_report.split('\n')
    
    lines = lines[2:]

    classes = []
    plotMat = []
    support = []
    class_names = []
    for line in lines[: number_of_classes]:
        t = list(filter(None, line.strip().split('  ')))
        if len(t) < 4: continue
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        support.append(int(t[-1]))
        class_names.append(t[0])
        plotMat.append(v)

    xlabel = 'Metrics'
    ylabel = 'Classes'
    xticklabels = ['Precision', 'Recall', 'F1-score']
    yticklabels = ['{0} ({1})'.format(class_names[idx], sup) for idx, sup  in enumerate(support)]
    figure_width = 10
    figure_height = len(class_names) + 3
    correct_orientation = True
    heatmap(np.array(plotMat), title, xlabel, ylabel, xticklabels, yticklabels, figure_width, figure_height, correct_orientation, cmap=cmap)
    plt.show()

plot_classification_report(classification_report=x)
st.pyplot()

sns.heatmap(df.corr(),cmap="coolwarm")
st.pyplot()

metrics.plot_roc_curve(clf, X_test, y_test) 
st.pyplot()

from sklearn.metrics import confusion_matrix
cf_matrix = confusion_matrix(y_test, y_pred)

ax = sns.heatmap(cf_matrix,annot=True,cmap='coolwarm')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')

ax.xaxis.set_ticklabels(['False','True'])
ax.yaxis.set_ticklabels(['False','True'])
st.pyplot()

