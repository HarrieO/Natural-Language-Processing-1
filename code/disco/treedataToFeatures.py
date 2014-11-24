#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil, csv
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
treeToIndices = []
treeIndex = 0
for tree, sentDict in result.items():
    treeToIndices.append(tree)
    #print '%3d\t%s' % (sum(b.values()), a)
    for key, count in sentDict.items():
        if tree in treeposts[indices[key]].fragments:
            treeposts[indices[key]].fragments[treeIndex]    += count
            featureMatrix[indices[key]][tree]               += count
        else:
            treeposts[indices[key]].fragments[treeIndex]    = count
            featureMatrix[indices[key]][tree]               = count
    treeIndex += 1
   
with open("featureData.csv", 'wb') as csvfile:
    writerObject = csv.writer(csvfile, delimiter=',')
    for post in treeposts:
        writerObject.writerow([post.id, post.content, post.score, post.community, post.fragments])  


file = open("indicesToTrees.txt", "w")
for i in range(treeIndex):
    print str(i),treeToIndices[i]
    file.write(str(i) + ' ')
    file.write(treeToIndices[i].encode("utf-8"))
    file.write('\n')

#[file.write((str(i) + ' ' + str(treeToIndices[i]) + u'\n').encode('utf8')) for i in range(treeIndex)]
file.close()

print "Added fragments to posts"

# Convert list of dicts to a sparse matrix
vectorizer = feature_extraction.DictVectorizer(sparse=True)
X = vectorizer.fit_transform(featureMatrix)
    
# Trivial machine learning objective: detect long sentences
target = []
for post in treeposts:
    if post.score > 0.5:
        target.append('polite')
    elif post.score > -0.5:
        target.append('neutral')
    else:
        target.append('impolite')
y = preprocessing.LabelEncoder().fit_transform(target)

# Use an SVM-like classifier and 10-fold crossvalidation for evaluation
classifier = linear_model.SGDClassifier(loss='hinge', penalty='elasticnet')
cv = cross_validation.StratifiedKFold(y, n_folds=10, shuffle=True, random_state=42)
print cross_validation.cross_val_score(classifier, X, y, cv=cv)