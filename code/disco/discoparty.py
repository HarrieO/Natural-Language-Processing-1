#!/usr/bin/env python2
# coding=utf-8
import glob
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation

vectorizer = feature_extraction.DictVectorizer(sparse=True)

text = treebank.BracketCorpusReader('../../datasets/successtrees.txt')
trees = [treetransforms.binarize(tree, horzmarkov=1, vertmarkov=1)
         for _, (tree, _) in text.itertrees(0, 1000)]
sents = [sent for _, (_, sent) in text.itertrees(0, 1000)]

result = fragments.getfragments(trees, sents, numproc=1, disc=False, cover=False)

for a, b in result.items()[:15]:
    print '%3d\t%s' % (sum(b.values()), a)