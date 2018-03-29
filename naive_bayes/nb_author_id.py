import sys
import os
from time import time
from tools.email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

"""
    this is the code to accompany the Lesson 1 (Naive Bayes) mini-project

    use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
features_train and features_test are the features for the training
and testing datasets, respectively
labels_train and labels_test are the corresponding item labels
"""
features_train, features_test, labels_train, labels_test = preprocess()

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print("training time:", round(time() - t0, 3), "s")
t1 = time()
pred = clf.predict(features_test)
print("predict time:", round(time() - t1, 3), "s")
print(accuracy_score(labels_test, pred))


"""
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 1.617 s
predict time: 0.263 s
0.973833902162
"""
