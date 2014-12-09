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
from sklearn.metrics import classification_report


'''
Settings
'''
N = 1000 # The amount of features selected for the histogram
forceExtractFeautres = False # If we want to extract features or not, otherwise load a Pickle file with preprocessed files, if the files are found
usePosTag = True

def getFeatures(trees, ignoredFeatures, features):
	global usePosTag
	results = list()
	i = 0
	for tree in trees:
		wordTags = getWordTagsFromTree(tree)
		if usePosTag:
			wordTags = [wordTag[0] for wordTag in wordTags]
		else:
			wordTags = [wordTag[0].split(" ")[1][:-1] for wordTag in wordTags]
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

	for N in [2000,4000,10000,20000]:
		print "============="
		print N
		print "============="
		fileTrainData 	= '../../datasets/preprocessed/baselineTrainSet'+str(N)+'.p'
		fileTestData	= '../../datasets/preprocessed/baselineTestSet'+str(N)+'.p'
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
		counts = {'neutral':0,'positive':0,'negative':0}
		maxCounts = trainClasses.count('negative')
		i = 0
		# Limit the class distribution
		# newTrainFeatures = list()
		# newTrainClasses = list()
		# while i<len(trainClasses):
		# 	if counts[trainClasses[i]]<maxCounts:
		# 		newTrainFeatures.append(trainFeatures[i])
		# 		newTrainClasses.append(trainClasses[i])
		# 		counts[trainClasses[i]]+=1
		# 	i += 1
		# print newTrainClasses[:10]
		# trainClasses = newTrainClasses
		# trainFeatures = newTrainFeatures
		print trainClasses.count('positive')
		print trainClasses.count('neutral')
		print trainClasses.count('negative')
		print trainFeatures[:10]
		vectorizer = feature_extraction.DictVectorizer(sparse=False)
		X     = vectorizer.fit_transform(trainFeatures)
		Xtest = vectorizer.transform(testFeatures)
		# gnb = naive_bayes.GaussianNB()
		# model = gnb.fit(X, trainClasses)
		# {'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960}
		#classifier = svm.SVC(kernel='rbf',gamma=0.1)
		classifier = svm.SVC(kernel='linear',class_weight={'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960})
		model = classifier.fit(X, trainClasses)


		print "Fit classifier, calculating scores"

		y_true = trainClasses
		y_pred = model.predict(X)
		print "Accuracy on training set:", metrics.accuracy_score(y_true, y_pred)
		cm = metrics.confusion_matrix(y_true, y_pred)
		print cm
		print(classification_report(y_true, y_pred, target_names=classes))

		y_true = testClasses
		y_pred = model.predict(Xtest)
		print "Accuracy on test set:    ", metrics.accuracy_score(y_true, y_pred)
		cm = metrics.confusion_matrix(y_true, y_pred)
		print cm
		#print y_pred
		print(classification_report(y_true, y_pred, target_names=classes))