#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil, csv, io
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
from subprocess import call
from treepost import *
from bracketStringReader import BracketStringReader

featureMap = {}
with open('indicesToTrees.txt') as f:
    for line in f:
        i, tree = line.split(' ',1)
        tree = tree.strip()
        featureMap[tree] = i

treeposts = read_posts('test.csv')

indices = []
with open('test_comment_indices.txt') as f:
    for line in f:
        indices.append(int(line))

vectorizer = feature_extraction.DictVectorizer(sparse=True)

treeStrings = [ line[:-1] for line in io.open('test_trees.txt', encoding='utf-8')]
print "Total of", len(treeStrings), "trees in test set."
treeStrings.extend(treeStrings[:])

text = BracketStringReader(treeStrings)
print "Made treebank"

trees = [treetransforms.binarize(tree, horzmarkov=1, vertmarkov=1)
         for _, (tree, _) in text.itertrees(0)]
print "Binarized trees"
sents = [sent for _, (_, sent) in text.itertrees(0)]

print "Starting fragment extraction"
result = fragments.getfragments(trees, sents, numproc=1, disc=False, cover=True)
print "Extracted fragments"

treeIndex = 0
found = 0
total = 0
for tree, sentDict in result.items():
    total += 1
    if tree in featureMap:
        found += 1
         #print '%3d\t%s' % (sum(b.values()), a)
        for key, count in sentDict.items():
            if key < len(indices):
                treeIndex = featureMap[tree.strip()]
                if treeIndex in treeposts[indices[key]].fragments:
                    treeposts[indices[key]].fragments[treeIndex]    += count
                else:
                    treeposts[indices[key]].fragments[treeIndex]    = count

print "Found", found, "out of", total,"trees in featurespace."

with open("testset.csv", 'wb') as csvfile:
    writerObject = csv.writer(csvfile, delimiter=',')
    writerObject.writerow(["id","content","score","community","features"]) 
    for post in treeposts:
        writerObject.writerow([post.id, post.content, post.score, post.community, post.fragments])  


print "Added fragments to posts"
