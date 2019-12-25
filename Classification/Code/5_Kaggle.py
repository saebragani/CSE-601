import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import random
import pdb
import statistics
import pandas as pd


def normalize_features(train, test):
    mean_train = np.mean(train, 0)
    mean_train_rep = np.repeat(mean_train.reshape(1,train.shape[1]), train.shape[0], axis=0)
    train_std = np.std(train, 0)
    train_std_rep = np.repeat(train_std.reshape(1, train.shape[1]), train.shape[0], axis=0)
    norm_train = (train - mean_train_rep) / train_std_rep

    mean_test = np.mean(test, 0)
    mean_test_rep = np.repeat(mean_test.reshape(1, test.shape[1]), test.shape[0], axis=0)
    test_std = np.std(test, 0)
    test_std_rep = np.repeat(test_std.reshape(1, test.shape[1]), test.shape[0], axis=0)
    norm_test = (test - mean_test_rep) / test_std_rep
    return norm_train, norm_test

def ensemble_majority(pred_knn, pred_nb, pred_lr, pred_rf, pred_dt):

    pred_knn=np.reshape(pred_knn,(-1,1))
    pred_svm = np.reshape(pred_nb, (-1, 1))
    pred_lr=np.reshape(pred_lr, (-1, 1))
    pred_rf=np.reshape(pred_rf, (-1, 1))
    pred_dt=np.reshape(pred_dt, (-1, 1))

    class_array=np.concatenate((pred_knn, pred_svm,pred_lr, pred_rf, pred_dt),axis=1)
    print(class_array)

    predicted_class = []
    for j in range(class_array.shape[0]):
        majority_class = statistics.mode(class_array[j,:])
        predicted_class.append(majority_class)
    return predicted_class

def bagging(X, y):
    sampled_train_ind = np.random.choice(X.shape[0], 8000*X.shape[0], replace=True)
    sampled_train = X[sampled_train_ind]
    sampled_labels = y[sampled_train_ind]
    return sampled_train, sampled_labels


if __name__ == '__main__':

    df1=pd.read_csv("./train_features.csv", header=None)
    X_train=(df1.values)[:,1:]

    df2 = pd.read_csv("./train_label.csv")
    y_train = (df2.values)[:,-1]

    X_train, y_train = bagging(X_train, y_train)

    df3 = pd.read_csv("./test_features.csv", header=None)
    X_test = (df3.values)[:,1:]

    #k nearest neighbour
    norm_train, norm_test = normalize_features(X_train, X_test)
    classifier = KNeighborsClassifier(n_neighbors=11)
    classifier.fit(norm_train, y_train)
    pred_knn = classifier.predict(norm_test)

    # # SVM implementation
    # svclassifier = SVC(kernel='linear')
    # svclassifier.fit(X_train, y_train)
    # pred_svm = svclassifier.predict(X_test)

    # Naive Bayes
    model_nb = GaussianNB()
    model_nb.fit(X_train, y_train)
    pred_nb = model_nb.predict(X_test)

    # Logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred_lr = model.predict(X_test)

    # Random forest
    model_rf = RandomForestClassifier(n_estimators=10, random_state=42)
    model_rf.fit(X_train, y_train)
    pred_rf = model_rf.predict(X_test)

    # Decision Tree
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train, y_train)
    pred_dt = model_dt.predict(X_test)

    ensemble_class=ensemble_majority(pred_knn, pred_nb, pred_lr, pred_rf, pred_dt)

    col1=np.arange(418,418+X_test.shape[0])
    dict1={"id":col1,"label":ensemble_class}

    tmp=pd.DataFrame(dict1)
    tmp.to_csv("./Submission.csv",index=False)