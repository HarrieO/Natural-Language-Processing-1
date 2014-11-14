#!/usr/bin/env python2
# coding=utf-8
import glob, pkgutil
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
vectorizer = feature_extraction.DictVectorizer(sparse=True)

def load_all(directory):
    for loader, name, ispkg in pkgutil.walk_packages([directory]):
        module = loader.find_module(name).load_module(name)
        exec('%s = module' % name)

if __name__ == '__main__':
	print 'Discodop test!!'

	load_all("discodop")

	dir(discodop)

	demos