#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil, csv
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm
import sklearn
from subprocess import call
from datapoint import *

print "Reading training data."

posts = read_data("featureData.csv")

print "Converting to feature matrix."

featureMatrix = [post.fragments for post in posts]

print "Vectorizing data."

# Convert list of dicts to a sparse matrix
vectorizer = feature_extraction.DictVectorizer(sparse=True)
X = vectorizer.fit_transform(featureMatrix)

print "Setting up target"

# Trivial machine learning objective: detect long sentences
target = []
for post in posts:
    if post.score > 0.5:
        target.append('polite')
    elif post.score > -0.5:
        target.append('neutral')
    else:
        target.append('impolite')
y = preprocessing.LabelEncoder().fit_transform(target)

print "Initiating cross validation"

# Use an SVM-like classifier and 10-fold crossvalidation for evaluation
classifier = svm.SVC(verbose=True)
cv = cross_validation.StratifiedKFold(y, n_folds=4, shuffle=True, random_state=42)
print cross_validation.cross_val_score(classifier, X.toarray(), y, cv=cv)