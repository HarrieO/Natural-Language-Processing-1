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

text = treebank.BracketCorpusReader('lesstrees.txt')
print "Made treebank"
trees = [treetransforms.binarize(tree, horzmarkov=1, vertmarkov=1)
         for _, (tree, _) in text.itertrees(0)]
print "Binarized trees"
sents = [sent for _, (_, sent) in text.itertrees(0)]

print "Starting fragment extraction"
result = fragments.getfragments(trees, sents, numproc=1, disc=False, cover=True)
print "Extracted fragments"
for tree, sentDict in result.items():
    #print '%3d\t%s' % (sum(b.values()), a)
    for key, count in sentDict.items():
        if tree in treeposts[indices[key]].fragments:
            treeposts[indices[key]].fragments[tree] += count
        else:
            treeposts[indices[key]].fragments[tree] = count
     
print "Added fragments to posts"
    

