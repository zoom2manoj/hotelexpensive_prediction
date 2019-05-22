import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
from pandas.plotting import scatter_matrix
from sklearn import datasets, linear_model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import classification_report, accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from sklearn.pipeline import Pipeline
from time import time
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import mpl_toolkits

train = pd.read_csv('train.csv')

train['encoded_spent'] = train['spent'].map({'high': 1, 'low': 0})

y = train.encoded_spent

# drop all string parameters for initial module
X = train.select_dtypes(include=[np.number]).interpolate().dropna()

X.columns.values

# split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=5)

# Creating cross validation data splits
cv_sets = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
cv_sets.get_n_splits(X_train, y_train)

## Initializing all models and parameters
# Initializing classifiers
RF_clf = RandomForestClassifier(n_estimators=200, random_state=1, class_weight='balanced')
AB_clf = AdaBoostClassifier(n_estimators=200, random_state=2)
GNB_clf = GaussianNB()
KNN_clf = KNeighborsClassifier()
LOG_clf = linear_model.LogisticRegression(multi_class="ovr", solver="sag", class_weight='balanced')
# clfs = [RF_clf, AB_clf, GNB_clf, KNN_clf, LOG_clf]
clfs = [AB_clf, GNB_clf, KNN_clf, LOG_clf]

n_jobs = 1  # Insert number of parallel jobs here

# Specficying scorer and parameters for grid search
feature_len = X.shape[1]
print(str(feature_len))
scorer = make_scorer(accuracy_score)
parameters_RF = {'clf__max_features': ['auto', 'log2'],
                 'dm_reduce__n_components': np.arange(5, feature_len, int(np.around(feature_len / 5)))}
parameters_AB = {'clf__learning_rate': np.linspace(0.5, 2, 5),
                 'dm_reduce__n_components': np.arange(5, feature_len, int(np.around(feature_len / 5)))}
parameters_GNB = {'dm_reduce__n_components': np.arange(5, feature_len, int(np.around(feature_len / 5)))}
parameters_KNN = {'clf__n_neighbors': [3, 5, 10],
                  'dm_reduce__n_components': np.arange(5, feature_len, int(np.around(feature_len / 5)))}
parameters_LOG = {'clf__C': np.logspace(1, 1000, 5),
                  'dm_reduce__n_components': np.arange(5, feature_len, int(np.around(feature_len / 5)))}

# parameters = {clfs[0]: parameters_RF, clfs[1]: parameters_AB, clfs[2]: parameters_GNB, clfs[3]: parameters_KNN, clfs[4]: parameters_LOG}
parameters = {clfs[0]: parameters_AB, clfs[1]: parameters_GNB, clfs[2]: parameters_KNN, clfs[3]: parameters_LOG}
print('hello')

## Training a baseline model and finding the best model composition using grid search
# Train a simple GBC classifier as baseline model
clf = KNN_clf
clf.fit(X_train, y_train)
print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
                                                     accuracy_score(y_train, clf.predict(X_train))))
print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__, accuracy_score(y_test, clf.predict(X_test))))
#
#
# def predict_labels(clf, best_pipe, features, target):
#     # Makes predictions using a fit classifier based on scorer. '''
#
#     # Start the clock, make predictions, then stop the clock
#     start = time()
#     y_pred = clf.predict(best_pipe.named_steps['dm_reduce'].transform(features))
#     end = time()
#
#     # Print and return results
#     print("Made predictions in {:.4f} seconds".format(end - start))
#     return accuracy_score(target.values, y_pred)
#
#
# ## Training a baseline model and finding the best model composition using grid search
# # Train a simple GBC classifier as baseline model
# print('starting pipeline....')
# feature_len = X.shape[1]
# print('1')
# clf = KNN_clf
# params = {'clf__max_features': ['auto', 'log2'],
#           'dm_reduce__n_components': np.arange(5, feature_len, int(np.around(feature_len / 5)))}
# jobs = 1
# print('2')
# scorer = make_scorer(accuracy_score)
# print('3')
# # cv_sets_test = model_selection.StratifiedShuffleSplit(n_splits=5, test_size=0.20, random_state=5)
# #
# # cv_sets_test.get_n_splits(X_test, y_test)
#
# pca = PCA()
# print('4')
# dm_reduction = pca
# print('5')
# # Define pipeline of dm reduction and classifier
# estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
# print('6')
# pipeline = Pipeline(estimators)
# print('7')
# train_prepared = pipeline.fit(X_train, y_train)
# print('8')
# test_prepared = pipeline.fit(X_test, y_test)
# print('9')
# print('hi')
# # Grid search over pipeline and return best classifier
# grid_obj = model_selection.GridSearchCV(clf, params, cv=5, scoring='neg_mean_squared_error', refit=True)
# print('hi')
# grid_obj.fit(X_train, y_train)
# print('hi')
# best_pipe = grid_obj.best_estimator_
# print(best_pipe)

#
# def train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs, use_grid_search=True,
#                      best_components=None, best_params=None):
#     # Fits a classifier to the training data. '''
#
#     # Start the clock, train the classifier, then stop the clock
#     start = time()
#
#     # Check if grid search should be applied
#     if use_grid_search == True:
#
#         # Define pipeline of dm reduction and classifier
#         estimators = [('dm_reduce', dm_reduction), ('clf', clf)]
#         pipeline = Pipeline(estimators)
#
#         # Grid search over pipeline and return best classifier
#         grid_obj = model_selection.GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv_sets, n_jobs=jobs)
#         try:
#             grid_obj.fit(X_train, y_train)
#         except Exception as e:
#             print(e)
#         best_pipe = grid_obj.best_estimator_
#     else:
#
#         # Use best components that are known without grid search
#         estimators = [('dm_reduce', dm_reduction(n_components=best_components)), ('clf', clf(best_params))]
#         pipeline = Pipeline(estimators)
#         best_pipe = pipeline.fit(X_train, y_train)
#
#     end = time()
#
#     # Print the results
#     print("Trained {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start) / 60))
#
#     # Return best pipe
#     return best_pipe
#
#
# def train_calibrate_predict(clf, dm_reduction, X_train, y_train, X_calibrate, y_calibrate, X_test, y_test, cv_sets,
#                             params, scorer, jobs, use_grid_search=True, **kwargs):
#     # Train and predict using a classifer based on scorer.
#
#     # Indicate the classifier and the training set size
#     print("Training a {} with {}...".format(clf.__class__.__name__, dm_reduction.__class__.__name__))
#
#     # Train the classifier
#     best_pipe = train_classifier(clf, dm_reduction, X_train, y_train, cv_sets, params, scorer, jobs)
#
#     #     #Calibrate classifier
#     #     print("Calibrating probabilities of classifier...")
#     #     start = time()
#     #     clf = CalibratedClassifierCV(best_pipe.named_steps['clf'], cv= 'prefit', method='isotonic')
#     #     clf.fit(best_pipe.named_steps['dm_reduce'].transform(X_calibrate), y_calibrate)
#     #     end = time()
#     #     print("Calibrated {} in {:.1f} minutes".format(clf.__class__.__name__, (end - start)/60))
#
#     # Print the results of prediction for both training and testing
#     print("Score of {} for training set: {:.4f}.".format(clf.__class__.__name__,
#                                                          predict_labels(clf, best_pipe, X_train, y_train)))
#     print("Score of {} for test set: {:.4f}.".format(clf.__class__.__name__,
#                                                      predict_labels(clf, best_pipe, X_test, y_test)))
#
#     # Return classifier, dm reduction, and label predictions for train and test set
#     return clf, best_pipe.named_steps['dm_reduce'], predict_labels(clf, best_pipe, X_train, y_train), predict_labels(
#         clf, best_pipe, X_test, y_test)
#
#
# def find_best_classifier(classifiers, dm_reductions, scorer, X_t, y_t, X_c, y_c, X_v, y_v, cv_sets, params, jobs):
#     # Tune all classifier and dimensionality reduction combiantions to find best classifier.
#
#     # Initialize result storage
#     clfs_return = []
#     dm_reduce_return = []
#     train_scores = []
#     test_scores = []
#
#     # Loop through dimensionality reductions
#     for dm in dm_reductions:
#
#         # Loop through classifiers
#         for clf in clfs:
#             # Grid search, calibrate, and test the classifier
#             clf, dm_reduce, train_score, test_score = train_calibrate_predict(clf=clf, dm_reduction=dm, X_train=X_t,
#                                                                               y_train=y_t, X_calibrate=X_c,
#                                                                               y_calibrate=y_c, X_test=X_v, y_test=y_v,
#                                                                               cv_sets=cv_sets, params=params[clf],
#                                                                               scorer=scorer, jobs=jobs,
#                                                                               use_grid_search=True)
#
#             # Append the result to storage
#             clfs_return.append(clf)
#             dm_reduce_return.append(dm_reduce)
#             train_scores.append(train_score)
#             test_scores.append(test_score)
#
#     # Return storage
#     return clfs_return, dm_reduce_return, train_scores, test_scores
#
#
# # Initializing dimensionality reductions
# pca = PCA()
# dm_reductions = [pca]
# scorer = make_scorer(accuracy_score)
#
# print('scorer', str(scorer))
#
# # Training all classifiers and comparing them
# X_calibrate = X_train
# y_calibrate = y_train
# clfs, dm_reductions, train_scores, test_scores = find_best_classifier(clfs, dm_reductions, scorer, X_train, y_train,
#                                                                       X_calibrate, y_calibrate, X_test, y_test, cv_sets,
#                                                                       parameters, n_jobs)
