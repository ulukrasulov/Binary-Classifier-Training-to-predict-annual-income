
from pandas import read_csv
from collections import Counter
import numpy as np
from numpy import mean
from numpy import std
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

#Training Dataset
direc = 'adult-train.csv'
df = read_csv(direc, na_values='?')
# missing rows dropped
df = df.dropna()

#class breakdown
y_var = df.values[:,-1]
counter = Counter(y_var)
for i,n in counter.items():
	percent = n / len(y_var) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (i, n, percent))
    

print(df.info)

print(df.dtypes)

"""Baseline evaluation of model through cross validation"""


# read data
def read_data(path):
    df = read_csv(direc, na_values='?')
    df = df.dropna()
    # split into inputs and outputs
    y_col = len(df.columns) - 1
    print("last_ix")
    print(y_col)
    print(df.iloc[:,13])
    print(df.drop(df.columns[y_col], axis=1))
    X, y = df.drop(df.columns[y_col], axis=1), df.iloc[:,y_col]
    # select categorical and numerical features
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    print(X.info)
    print(X.columns)
    return X.values, y


mean_array=np.array([])
std_arrya=np.array([])

# define the location of the dataset

# load the dataset
X, y = read_data(direc)
# summarize the loaded dataset
print(X.shape, y.shape, Counter(y))

"""Dummy Classifier"""

# Select Model
model = DummyClassifier(strategy='most_frequent')
# Calculate stratified cross validation score 
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Dummy Classifier Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

mean_array=np.append(mean_array,mean(scores))
std_arrya=np.append(std_arrya,std(scores))








X, y = read_data(direc)

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
#Transform categorical variables
for i in range(13):
    X[:,i] = le.fit_transform(X[:,i])





"""Support Vector Machine"""
model = SVC(gamma='scale')
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Support Vector Mchine Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
mean_array=np.append(mean_array,mean(scores))
std_arrya=np.append(std_arrya,std(scores))

"""BaggingClassifier"""


model = BaggingClassifier(n_estimators=100)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('BaggingClassifier Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
mean_array=np.append(mean_array,mean(scores))
std_arrya=np.append(std_arrya,std(scores))


"""RandomForestClassifier"""

model = RandomForestClassifier(n_estimators=100)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
print('Random Forest Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
mean_array=np.append(mean_array,mean(scores))
std_arrya=np.append(std_arrya,std(scores))



"""GradientBoostingClassifier""" 
    
model = GradientBoostingClassifier(n_estimators=100)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)


print('Gradient Boosting: %.3f (%.3f)' % (mean(scores), std(scores)))
mean_array=np.append(mean_array,mean(scores))
std_arrya=np.append(std_arrya,std(scores))


print(mean_array)
print(std_arrya)

model.fit(X,y)

"""Plot different models results"""

plt.rcParams['figure.figsize']=(12,18)
plt.errorbar(["Dummy Classifier","Support Vector Machine","Bagging Classifier","Random Forest Classifier" ,"Gradient Boosting Classifier"],mean_array,yerr=std_arrya,fmt='o', elinewidth=3)
plt.ylabel('Cross Validation Score', fontsize=14)
plt.show()


"""SELECT OUR MAIN MODEL AS GRADIENT BOOSTER"""
model = GradientBoostingClassifier(n_estimators=100).fit(X,y)
test_path = 'adult-test.csv'


X_test, y_test = read_data(test_path)

#Transform categorical variables
for i in range(13):
    X_test[:,i] = le.fit_transform(X_test[:,i])
    
    

#Predicted labels
y_predicted = model.predict(X_test)
print(y_predicted)
"""
predict_proba calculates the predicted probabilitity of the given classes for y_predicted: """

y_predict_prob = model.predict_proba(X_test)
print(y_predict_prob)

#################
#################
##EVALUATION
#################
#################

""" Jaccard Index for Evaluation 
"""

from sklearn.metrics import jaccard_score
jaccard_evaluation=jaccard_score(y_test, y_predicted)
print("JACCARD EVALUATION")
print(jaccard_evaluation)

""" Log loss Evaluation"""
 
from sklearn.metrics import log_loss
log_loss_evaluation=log_loss(y_test, y_predict_prob)
print("LOG LOSS EVALUATION")
print(log_loss_evaluation)

"""Confusion Matrix Evaluation"""

from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, y_predicted, labels=[1,0]))
print (classification_report(y_test, y_predicted))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_predicted, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=["Above 50k","Below 50k"],normalize= False,  title='Confusion matrix')
plt.show()