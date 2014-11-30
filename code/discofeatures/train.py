#!/usr/bin/env python2
# coding=utf-8
import numpy as np
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes
import sklearn
from datapoint import *

print "Reading training data."

training = read_data("trainset.csv")
test     = read_data("testset.csv")

print "Converting to feature matrix."

featureMatrix = [post.fragments for post in training]

testMatrix = [post.fragments for post in training]

print "Vectorizing data."

# Convert list of dicts to a sparse matrix
vectorizer = feature_extraction.DictVectorizer(sparse=False)
X = vectorizer.fit_transform(featureMatrix)

Xtest = vectorizer.transform(testMatrix)

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

labelEncoder = preprocessing.LabelEncoder()

# train targets
y = labelEncoder.fit_transform(target)
# true values of test data
r = labelEncoder.transform(real)


print "Initiating cross validation"

classifier = naive_bayes.GaussianNB()
classifier.fit(X, y)

print "Fit classifier, calculating scores"

print "Accuracy on training set:", classifier.score(X,y)
print "Accuracy on test set:    ", classifier.score(Xtest,r)