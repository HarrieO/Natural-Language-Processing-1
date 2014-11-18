#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil, csv
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
from subprocess import call

# In order to import the main project code
sys.path.append('..')
# import the data reader
import post 

table = post.read_table('../train.csv')

def write_csv(data,fileName):
    with open(fileName, 'wb') as csvfile:
        writerObject = csv.writer(csvfile, delimiter=',')
        for row in data:
            writerObject.writerow(row)

with open('trees.txt') as f1:
    with open('sentences.txt') as f2:
        spaces = 3
        for line in f2:
            # there are never more than 3 newlines in the file
            # so this is a check for EoF
            while spaces > 0:
                toAdd = f1.readline()
                if toAdd[:1] == "(":
                    print line, toAdd
                    print
                    break
                elif toAdd == "":
                    spaces -= 1

exit()

treeList = []
with open('trees.txt') as f1:
    with open('indices.txt') as f2:
        curIndex = 0
        trees = ""
        spaces = 3
        for line in f2:
            index = int(line)
            if index != curIndex:
                treeList += [trees.strip()]
                curIndex = index
                trees = ""
            # there are never more than 3 newlines in the file
            # so this is a check for EoF
            while spaces > 0:
                toAdd = f1.readline()
                if toAdd[:1] == "(":
                    spaces = 3
                    trees += " " + toAdd
                    break
                elif toAdd == "":
                    spaces -= 1
        treeList += [trees.strip()]

print len(table), len(treeList)
result = []
for i,row in enumerate(table):
    result += [[i, " ".join(row[0].split())] + row[1:] + [" ".join(treeList[i].split())]]

write_csv(result,"discotrain.csv")


#The tokenizer which splits sentences
def split_sentences(txt):
	return (re.findall(r'(?ms)\s*(.*?(?:\.|\?|!))', txt))  # split sentences

# Read the comments in.
contents    = post.read_column(0,'../train.csv')

