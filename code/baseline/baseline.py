import re
import os, sys
import numpy as np
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
import post
from decisionstumps import reduceFeatureSpace, outputHistograms, emptyClassCount
from sklearn.naive_bayes import GaussianNB

'''
Settings
'''
N = 1000 # The amount of features selected for the histogram

if __name__ == "__main__":
	# Data structures
	classes			= ['negative','neutral','positive']
	classCutOff		= [-0.5,0.5]
	classCount 		= emptyClassCount(classes)
	wordTagCount 	= dict()
	totalScores		= dict()


	# Running starts here
	
	# First extract the counts
	newCounts, ignoredWordTags = reduceFeatureSpace(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'),classes,classCutOff,wordTagCount,classCount,totalScores, N);

	# Extract the histograms based on the selected features
	outputHistograms(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'), os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/trainHist.csv'), classes, classCutOff, newCounts.keys());
	#outputHistograms(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/test_trees.txt'), os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/testHist.csv'), classes, classCutOff, newCounts.keys(), True);

	# Load the histograms back in

	# These counts are used for training the baseline classifier
	gnb = GaussianNB()
	model = gnb.fit(iris.data, iris.target)


