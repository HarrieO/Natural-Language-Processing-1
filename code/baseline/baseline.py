import re
import os, sys
import numpy as np
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import post
import extractFeatures
from decisionstumps import *
from sklearn import *
sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))
from treedataToJoosttrees import getPostsWithTrees


'''
Settings
'''
N = 5000 # The amount of features selected for the histogram
forceExtractFeautres = False # If we want to extract features or not, otherwise load a Pickle file with preprocessed files, if the files are found
usePosTag = False

def getFeatures(trees, ignoredFeatures, features):
	global usePosTag
	results = list()
	i = 0
	for tree in trees:
		wordTags = getWordTagsFromTree(tree)
		if usePosTag:
			wordTags = [wordTag[0] for wordTag in wordTags]
		else:
			wordTags = [wordTag[0].split(" ")[:-1] for wordTag in wordTags]
		# results.append(extractFeatures.extract_features_word(wordTags, ignoredFeatures, features))
		results.append(dict(extractFeatures.extract_features_word(wordTags, ignoredFeatures, features)))
		# print str(float(i)/float(len(trees)))
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
	testData = getPostsWithTrees('../../datasets/preprocessed/')
	
	if os.path.isfile(fileTrainData) and os.path.isfile(fileTestData) and not forceExtractFeautres:
		# Skip file extraction if preprocessed files are available
		trainFeatures = pickle.load(open(fileTrainData,'rb'))
		testFeatures = pickle.load(open(fileTestData,'rb'))
	else:
		# First extract the counts
		counts, ignoredWordTags = reduceFeatureSpace(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'),classes,classCutOff,wordTagCount,classCount,totalScores, N)

		# Extract the histograms based on the selected features
		#outputHistograms(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'), os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/trainHist.csv'), classes, classCutOff, counts.keys())
	
		treesTrain = post.read_column(4,os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'))

		trainFeatures = getFeatures(treesTrain,ignoredWordTags,counts.keys())
		# treesTest = open(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/test_trees.txt'), 'r')
		treesTest = [" ".join(row.trees) for row in testData]
		testFeatures = getFeatures(treesTest,ignoredWordTags,counts.keys())

		print np.shape(trainFeatures[0])
		pickle.dump(trainFeatures, open(fileTrainData,'w+b'))
		pickle.dump(testFeatures, open(fileTestData,'w+b'))

	trainScores = [float(row) for row in post.read_column(1,'../../datasets/preprocessed/train.csv')]
	testScores = [row.score for row in testData]
	# testScores = post.read_column(1,'../../datasets/preprocessed/test.csv')
	trainClasses = scoresToClass(trainScores,classCutOff,classes)
	testClasses = scoresToClass(testScores,classCutOff,classes)


	vectorizer = feature_extraction.DictVectorizer(sparse=False)
	X     = vectorizer.fit_transform(trainFeatures)
	Xtest = vectorizer.transform(testFeatures)
	gnb = GaussianNB()
	print "X"
	print np.shape(X)
	print len(trainClasses)
	print "Xtest"
	print np.shape(Xtest)
	print len(testClasses)
	model = gnb.fit(X, trainClasses)
	# classifier = svm.SVC()
	# model = classifier.fit(X, trainClasses)


	print "Fit classifier, calculating scores"

	print "Accuracy on training set:", model.score(X,trainClasses)
	print "Accuracy on test set:    ", model.score(Xtest,testClasses)