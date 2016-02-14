#!/usr/bin/python

import sys
import pickle
import numpy as np

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
from operator import itemgetter
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

# Utility function to report best scores
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


### Task 1
### Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'exercised_stock_options',
                 'total_stock_value',
                 'deferred_income',
                 'annual_cash' ## new feature salary + bonus
                ]

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2
### Remove outliers
## TOTAL key to be dropped
print "removing outlier: TOTAL"
data_dict.pop('TOTAL', 0)

## this company could have been involved in the fraud but we are interested in actual people
print "removing outlier: THE TRAVEL AGENCY IN THE PARK"
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0)

## lets remove people with too much NaN values. We cannot say they are POI if no info about them
NAN_LIMIT = 17
nanCounter = {}
for key in data_dict:
    nanCounter[key] = 0
    for feature in data_dict[key]:
        if data_dict[key][feature] == "NaN":
            nanCounter[key] = nanCounter[key] + 1

for counterKey in nanCounter :
    if (nanCounter[counterKey] > NAN_LIMIT) and not data_dict[counterKey]["poi"]:
        print "removing outlier:", counterKey
        data_dict.pop(counterKey, 0)

### Task 3
### Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
for key in my_dataset:
    bonus = my_dataset[key]['bonus']
    salary = my_dataset[key]['salary']
    if bonus != 'NaN' and salary != 'Nan':
        cash = salary + bonus
        my_dataset[key]['annual_cash'] = cash
    else:
        my_dataset[key]['annual_cash'] = 'NaN'


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

## feature scaling
minmax_scaler = MinMaxScaler()
features = minmax_scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

naive_bayes_clf = GaussianNB()
naive_bayes_clf.fit(features,labels)

random_forest_clf = RandomForestClassifier()
random_forest_clf.fit(features,labels)

decision_tree_clf = DecisionTreeClassifier()
decision_tree_clf.fit(features, labels)

adaboost_rf_clf = AdaBoostClassifier(base_estimator=random_forest_clf)
adaboost_rf_clf.fit(features,labels)

adaboost_dt_clf = AdaBoostClassifier(base_estimator=decision_tree_clf)
adaboost_dt_clf.fit(features,labels)



### Task 5
### Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
test_classifier(naive_bayes_clf, my_dataset, features_list)
#test_classifier(random_forest_clf, my_dataset, features_list)
#test_classifier(decision_tree_clf, my_dataset, features_list)
#test_classifier(adaboost_rf_clf, my_dataset, features_list)
#test_classifier(adaboost_dt_clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
clf = naive_bayes_clf
dump_classifier_and_data(clf, my_dataset, features_list)