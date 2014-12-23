import os, sys, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import post
import time, re, os, sys, operator
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))

sys.path.append('../../disco')
#import datapoint
import extractFeatures
from decisionstumps import *
from treedataToJoosttrees import getPostsWithTrees
#from sklearn.metrics import *
#import sklearn
#from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes, decomposition, neighbors
#from sklearn.externals import joblib




classes			= ['negative','neutral','positive']
classCutOff		= [-0.5,0.5]
classCount 		= emptyClassCount(classes)
wordTagCount 	= dict()
totalScores		= dict()
usePosTags      = False

testData = getPostsWithTrees('../../datasets/preprocessed/')




# First extract the counts
#counts, ignoredWordTags = reduceFeatureSpace(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'),classes,classCutOff,wordTagCount,classCount,totalScores, features, usePosTags)
#treesTrain = post.read_column(4,os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'))
#trainFeatures = getFeatures(treesTrain,ignoredWordTags,counts.keys(), usePosTags)

#treesTest = [" ".join(row.trees) for row in testData]
#testFeatures = getFeatures(treesTest,ignoredWordTags,counts.keys(), usePosTags)

########################################## CODE BY TIES ################################
extract('../../datasets/preprocessed/discotrain.csv',classes,classCutOff,wordTagCount,classCount,totalScores,usePosTags);
wordTag_entropy = word_entropy(classCount, wordTagCount)
print type(wordTag_entropy)
entropy_sorted_words=wordTag_entropy;
print len(entropy_sorted_words)

writer = csv.writer(open('../../datasets/preprocessed/word_entropy2.csv', 'w+b'))
for i in sorted(wordTag_entropy, key=wordTag_entropy.get, reverse=True):
	writer.writerow([i, wordTag_entropy[i]])
#for key, value in entropy_sorted_words.items():
#   writer.writerow([key, value])

#trainFeatures
#testFeatures