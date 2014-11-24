#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
from subprocess import call
from treepost import *


treeposts = read_posts('discotrain.csv')

indices = []
with open('indices.txt') as f:
    for line in f:
        indices.append(int(line))

vectorizer = feature_extraction.DictVectorizer(sparse=True)

text = treebank.BracketCorpusReader('trees.txt')
print "Made treebank"
trees = [treetransforms.binarize(tree, horzmarkov=1, vertmarkov=1)
         for _, (tree, _) in text.itertrees(0)]
print "Binarized trees"
sents = [sent for _, (_, sent) in text.itertrees(0)]

print "Starting fragment extraction"
result = fragments.getfragments(trees, sents, numproc=1, disc=False, cover=True)
print "Extracted fragments"
featureMatrix = [{} for _ in range(len(treeposts))]
for tree, sentDict in result.items():
    #print '%3d\t%s' % (sum(b.values()), a)
    for key, count in sentDict.items():
        if tree in treeposts[indices[key]].fragments:
            treeposts[indices[key]].fragments[tree] += count
            featureMatrix[indices[key]][tree] += count
        else:
            treeposts[indices[key]].fragments[tree] = count
            featureMatrix[indices[key]][tree] = count
     
print "Added fragments to posts"

# Convert list of dicts to a sparse matrix
vectorizer = feature_extraction.DictVectorizer(sparse=True)
X = vectorizer.fit_transform(featureMatrix)
    
# Trivial machine learning objective: detect long sentences
target = ['polite' if post.score > 0 else 'impolite' for post in treeposts]
y = preprocessing.LabelEncoder().fit_transform(target)

# Use an SVM-like classifier and 10-fold crossvalidation for evaluation
classifier = linear_model.SGDClassifier(loss='hinge', penalty='elasticnet')
cv = cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True, random_state=42)
print cross_validation.cross_val_score(classifier, X, y, cv=cv)