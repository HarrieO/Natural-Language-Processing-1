import re
import os, sys
import numpy as np
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import post
import extractFeatures
from decisionstumps import *
from sklearn.naive_bayes import GaussianNB

'''
Settings
'''
N = 1000 # The amount of features selected for the histogram
forceExtractFeautres = False # If we want to extract features or not, otherwise load a Pickle file with preprocessed files, if the files are found

def getFeatures(trees, ignoredFeatures, features):
	results = list()
	i = 0
	for tree in trees:
		wordTags = getWordTagsFromTree(tree)
		wordTags = [wordTag[0] for wordTag in wordTags]
		# results.append(extractFeatures.extract_features_word(wordTags, ignoredFeatures, features))
		extractFeatures.extract_features_word(wordTags, ignoredFeatures, features)
		# if (i % 1000) == 0:
		# 	print i
		i += 1
	return results


if __name__ == "__main__":
	# Data structures
	classes			= ['negative','neutral','positive']
	classCutOff		= [-0.5,0.5]
	classCount 		= emptyClassCount(classes)
	wordTagCount 	= dict()
	totalScores		= dict()
	fileTrainData 	= '../../datasets/preprocessed/baselineTrainSet.p'
	fileTestData	= '../../datasets/preprocessed/baselineTestSet.p'

	# Running starts here
	
	if os.path.isfile(fileTrainData) and os.path.isfile(fileTestData) and not forceExtractFeautres:
		# Skip file extraction if preprocessed files are available
		trainFeatures = pickle.dump(open(fileTrainData,'rb'))
		testFeatures = pickle.dump(open(fileTestData,'rb'))
	else:
		# First extract the counts
		counts, ignoredWordTags = reduceFeatureSpace(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'),classes,classCutOff,wordTagCount,classCount,totalScores, N)

		# Extract the histograms based on the selected features
		#outputHistograms(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'), os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/trainHist.csv'), classes, classCutOff, counts.keys())
	
		treesTrain = post.read_column(4,os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'))
		trainFeatures = getFeatures(treesTrain,ignoredWordTags,counts.keys())
		treesTest = open(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/test_trees.txt'), 'r')
		testFeatures = getFeatures(treesTest,ignoredWordTags,counts.keys())

		pickle.dump(trainFeatures, open(fileTrainData,'w+b'))
		pickle.dump(testFeatures, open(fileTestData,'w+b'))

	trainScores = post.read_column(1,'../../datasets/preprocessed/train.csv')
	testScores = post.read_column(1,'../../datasets/preprocessed/test.csv')
	trainClasses = scoresToClass(trainScores,classCutOff,classes)
	testClasses = scoresToClass(testScores,classCutOff,classes)
	gnb = GaussianNB()
	model = gnb.fit(trainFeatures, trainClasses)


