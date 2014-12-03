#!/usr/bin/env python2
# coding=utf-8
import gc
import numpy as np
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes, decomposition, tree
import sklearn
from featureDeduction import FeatureDeduction
from datapoint import *

print "Reading training data."

training = read_data("../../datasets/preprocessed/trainset.csv")
test     = read_data("../../datasets/preprocessed/testset.csv")

print "Converting to feature matrix."

deduct = FeatureDeduction(200)

featureMatrix = [deduct.featureDeduct(post.fragments) for post in training]
testMatrix = [deduct.featureDeduct(post.fragments) for post in test]

# featureMatrix = [post.fragments for post in training]
# testMatrix    = [post.fragments for post in test]


print "Vectorizing data."

# Convert list of dicts to a sparse matrix
vectorizer = feature_extraction.DictVectorizer(sparse=False)
X = vectorizer.fit_transform(featureMatrix)

Xtest = vectorizer.transform(testMatrix)

featureMatrix = None
testMatrix = None

print "Setting up target"

def giveLabel(score):
    if post.score > 0.5:
        return 'polite'
    elif post.score > -0.5:
        return 'neutral'
    else:
        return 'impolite'

# Trivial machine learning objective: detect long sentences
target = []
for post in training:
    target.append(giveLabel(post.score))
real = []
for post in test:
    real.append(giveLabel(post.score))

training = None
test = None

labelEncoder = preprocessing.LabelEncoder()

# train targets
y = labelEncoder.fit_transform(target)
# true values of test data
r = labelEncoder.transform(real)


print "Fitting classifier"

classifier = ensemble.AdaBoostClassifier()
classifier.fit(X, y)

print "Fit classifier, calculating scores"

print "Accuracy on test set:    ", classifier.score(Xtest,r)
print "Accuracy on training set:", classifier.score(X,y)