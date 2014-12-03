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
        if treeIndex in treeposts[indices[key]].fragments:
            treeposts[indices[key]].fragments[treeIndex]    += count
            featureMatrix[indices[key]][tree]               += count
        else:
            treeposts[indices[key]].fragments[treeIndex]    = count
            featureMatrix[indices[key]][tree]               = count
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

print "Added fragments to posts"