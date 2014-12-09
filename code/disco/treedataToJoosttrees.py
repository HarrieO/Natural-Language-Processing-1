#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil, csv
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
from subprocess import call
from treepost import *

def getPostsWithTrees(preprocessed_path):
    treeposts = read_posts(preprocessed_path + 'test.csv')

    indices = []
    with open(preprocessed_path + 'test_indices.txt') as f:
        for line in f:
            indices.append(int(line))

    trees = []
    with open(preprocessed_path + 'test_trees.txt') as f:
        for line in f:
            trees.append(line[:-1])

    for i, tree in enumerate(trees):
        print i, len(indices)
        post_i   = indices[i]
        treepost = treeposts[post_i]
        treepost.trees.append(tree) 

    return treeposts