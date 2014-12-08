#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil, csv
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
from subprocess import call
from treepost import *


treeposts = read_posts('../../datasets/preprocessed/discotrain.csv')

indices = []
with open('../../datasets/preprocessed/indices.txt') as f:
    for line in f:
        indices.append(int(line))

vectorizer = feature_extraction.DictVectorizer(sparse=True)

text = treebank.BracketCorpusReader('../../datasets/preprocessed/trees.txt')
print "Made treebank"
trees = [treetransforms.binarize(tree, horzmarkov=1, vertmarkov=1)
         for _, (tree, _) in text.itertrees(0)]
print "Binarized trees"
sents = [sent for _, (_, sent) in text.itertrees(0)]

print "Starting fragment extraction"
result = fragments.getfragments(trees, sents, numproc=1, disc=False, cover=True)
print "Extracted fragments"
featureMatrix    = [{} for _ in range(len(treeposts))]
treeToIndices    = []
averageTreeScore = []
numberOfPosts    = []
numberOfPolite   = []
numberOfImpolite = []
treeIndex = 0
for tree, sentDict in result.items():
    treeToIndices.append(tree)
    averageTreeScore.append(0)
    numberOfPosts.append(0)
    numberOfPolite.append(0)
    numberOfImpolite.append(0)
    #print '%3d\t%s' % (sum(b.values()), a)
    for key, count in sentDict.items():
        if treeIndex in treeposts[indices[key]].fragments:
            treeposts[indices[key]].fragments[treeIndex]    += count
            featureMatrix[indices[key]][tree]               += count
        else:
            treeposts[indices[key]].fragments[treeIndex]    = count
            featureMatrix[indices[key]][tree]               = count
            averageTreeScore[-1] += treeposts[indices[key]].score
            numberOfPosts[-1] += 1
            if treeposts[indices[key]].score > 0.5:
                numberOfPolite[-1] += 1
            elif treeposts[indices[key]].score < -0.5:
                numberOfImpolite[-1] += 1

    averageTreeScore[-1] /= numberOfPosts[-1]
    treeIndex += 1
   
with open("../../datasets/preprocessed/trainset.csv", 'wb') as csvfile:
    writerObject = csv.writer(csvfile, delimiter=',')
    writerObject.writerow(["id","content","score","community","features"]) 
    for post in treeposts:
        writerObject.writerow([post.id, post.content, post.score, post.community, post.fragments])  


file = open("../../datasets/preprocessed/indicesToTrees.txt", "w")
for i in range(treeIndex):
    file.write(str(i) + ' ')
    file.write(treeToIndices[i].encode("utf-8"))
    file.write('\n')
#[file.write((str(i) + ' ' + str(treeToIndices[i]) + u'\n').encode('utf8')) for i in range(treeIndex)]
file.close()

file = open("../../datasets/preprocessed/indicesToAverageScore.txt", "w")
[file.write(str(i) + ' ' + str(averageTreeScore[i]) + ' ' + str(numberOfPosts[i]) + ' ' + str(numberOfPolite[i]) + ' ' +
 str(numberOfImpolite[i]) +'\n') for i in range(treeIndex)]
file.close()

print "Added fragments to posts"