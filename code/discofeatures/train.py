#!/usr/bin/env python2
# coding=utf-8
import gc
import numpy as np
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes, decomposition
import sklearn
from datapoint import *

print "Reading training data."

training = read_data("trainset.csv")
test     = read_data("testset.csv")

print "Converting to feature matrix."

featureMatrix = [post.fragments for post in training]

testMatrix = [post.fragments for post in test]

print "Vectorizing data."

# Convert list of dicts to a sparse matrix
vectorizer = feature_extraction.DictVectorizer(sparse=False)
X = vectorizer.fit_transform(featureMatrix)

Xtest = vectorizer.transform(testMatrix)

featureMatrix = None
testMatrix = None

gc.collect()

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

gc.collect()

labelEncoder = preprocessing.LabelEncoder()

# train targets
y = labelEncoder.fit_transform(target)
# true values of test data
r = labelEncoder.transform(real)


print "Fitting classifier"

classifier = naive_bayes.GaussianNB()
classifier.fit(X, y)

print "Fit classifier, calculating scores"

# correct = 0
# total   = 0
# for x,t in zip(X,y):
#     p = classifier.predict(x)
#     if p == t:
#         correct += 1
#         print p, t
#     total += 1.0
#     print "Accuracy", (correct/total), "with", correct, "out of", total



print "Accuracy on test set:    ", classifier.score(Xtest,r)
print "Accuracy on training set:", classifier.score(X,y)