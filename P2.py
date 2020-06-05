import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
from sklearn import preprocessing
import sklearn.model_selection as ms
from sklearn import linear_model
import sklearn.metrics as sklm

import warnings
warnings.filterwarnings("ignore")


bikes = pd.read_csv('AdvWorksCustsB.csv')
#print(bikes.shape)
#bikes.head()

bikes_counts = bikes[['LastName', 'Bike Buyer']].groupby('Bike Buyer').count()
#print(bikes_counts)

labels = np.array(bikes['Bike Buyer'])

def encode_string(cat_features):
    ## Encoding strings to numeric categories
    enc = preprocessing.LabelEncoder()
    enc.fit(cat_features)
    enc_cat_features = enc.transform(cat_features)
    ## Applying one hot encoding
    ohe = preprocessing.OneHotEncoder()
    encoded = ohe.fit(enc_cat_features.reshape(-1,1))
    return encoded.transform(enc_cat_features.reshape(-1,1)).toarray()

categorical_columns = ['StateProvinceName', 'CountryRegionName',
                       'Education','Occupation','Gender','MaritalStatus']

Features = encode_string(bikes['City'])
for col in categorical_columns:
    temp = encode_string(bikes[col])
    Features = np.concatenate([Features, temp], axis = 1)


#print(Features.shape)
#print(Features[:2, :])
Features = np.concatenate([Features, np.array(bikes[['Age', 'CustomerID',
                            'HomeOwnerFlag', 'NumberCarsOwned','NumberChildrenAtHome','TotalChildren','YearlyIncome']])], axis = 1)
#print(Features.shape)
#print(Features[:2, :])

## Randomly sample cases to create independent training and test data
nr.seed(9988)
indx = range(Features.shape[0])
indx = ms.train_test_split(indx, test_size = 300)
X_train = Features[indx[0],:]
y_train = np.ravel(labels[indx[0]])
X_test = Features[indx[1],:]
y_test = np.ravel(labels[indx[1]])

scaler = preprocessing.StandardScaler().fit(X_train[:,34:])
X_train[:,34:] = scaler.transform(X_train[:,34:])
X_test[:,34:] = scaler.transform(X_test[:,34:])
X_train[:2,]

logistic_mod = linear_model.LogisticRegression()
logistic_mod.fit(X_train, y_train)

#print(logistic_mod.intercept_)
#print(logistic_mod.coef_)

probabilities = logistic_mod.predict_proba(X_test)
#print(probabilities[:15,:])

def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,1]])
scores = score_model(probabilities, 0.5)
print(np.array(scores[:15]))
print(y_test[:15])

def print_metrics(labels, scores):
    metrics = sklm.precision_recall_fscore_support(labels, scores)
    conf = sklm.confusion_matrix(labels, scores)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])
    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])
    print('')
    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))
    print(' ')
    print('           Positive      Negative')
    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])
    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])
    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])
    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])



print_metrics(y_test, scores)

def plot_auc(labels, probs):
    ## Compute the false positive rate, true positive rate
    ## and threshold along with the AUC
    fpr, tpr, threshold = sklm.roc_curve(labels, probs[:,1])
    auc = sklm.auc(fpr, tpr)

    ## Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color = 'orange', label = 'AUC = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

plot_auc(y_test, probabilities)

probs_positive = np.concatenate((np.ones((probabilities.shape[0], 1)),
                                 np.zeros((probabilities.shape[0], 1))),
                                 axis = 1)
scores_positive = score_model(probs_positive, 0.5)
print_metrics(y_test, scores_positive)
plot_auc(y_test, probs_positive)
