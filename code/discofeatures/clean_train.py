#!/usr/bin/env python2
# coding=utf-8
import os, sys, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
import gc
import numpy as np
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes, decomposition
import sklearn
from featureDeduction import FeatureDeduction
from datapoint import *

def feature2vector(train_data,test_data,feature_deduction=None,returnVectorizer = False):
	"""
	Takes train and test_data data as read from the csv file.
	Returns X and Xtest_data, sparse matrices.
	If set, feature deduction is done as well
	"""

	if feature_deduction:
		print "Deducing features to {0}".format(feature_deduction)
		deduct = FeatureDeduction(feature_deduction)

		print "Converting to feature matrix."
		featureMatrix = [deduct.featureDeduct(post.fragments) for post in train_data]
		testMatrix = [deduct.featureDeduct(post.fragments) for post in test_data]
	else:

		print "Converting to feature matrix."
		featureMatrix = [post.fragments for post in train_data]
		testMatrix = [post.fragments for post in test_data]

	print "Vectorizing data."

	# Convert list of dicts to a sparse matrix
	vectorizer = feature_extraction.DictVectorizer(sparse=False)
	X     = vectorizer.fit_transform(featureMatrix)
	Xtest = vectorizer.transform(testMatrix)

	# Delete the large matrices
	featureMatrix = None
	testMatrix = None

	if returnVectorizer:
		return X, Xtest, vectorizer
	else:
		return X, Xtest

def getLabels(training_data, test_data,splitPoints=[],splitProportions=[0.25, 0.25],verbose=True,returnEncoder=False):
	"""
	Returns labels for the data, splits data into proportions given by 
	splitProportions = [propNegative, propPositive]
	of absolute values given by splitPoints = [Negativepoint, Positivepoint]
	"""


	if not splitPoints:
		# Calculate split points on the basis of proportions
		t = [post.score for post in training_data]
		t.sort()
		ind1 = int(round(         len(t) * splitProportions[0]))
		ind2 = int(round(len(t) - len(t) * splitProportions[1]))
		splitPoints = [t[ind1],t[ind2]]

	if verbose: print "Split points:"
	if verbose: print splitPoints

	def giveLabel(score):
		if   post.score < splitPoints[0]:
			return 'impolite'
		elif post.score >= splitPoints[1]:
			return 'polite'
		else:
			return 'neutral'

	target = [giveLabel(post.score) for post in training_data]
	real   = [giveLabel(post.score) for post in test_data]

	labelEncoder = preprocessing.LabelEncoder()

	# train targets
	y = labelEncoder.fit_transform(target)
	# true values of test data
	r = labelEncoder.transform(real)

	if returnEncoder:
		return y,r, labelEncoder
	else:
		return y,r

def print_lbl_dist(y):
	for l in set(y):
		print "{0}:{1}".format(l, 1.0 * sum([1 for p in y if p==l])/len(y))


def getProcessedData(method=2,returnIndices=False):
	"""
	Loads and labels the data in according to the selected method:
	M0: 2lbls equal split
	M1: 2lbls [0.25 0.75 0.25] split, discard neutral
	M2: 3lbl [0.25 0.75 0.25] split
	M3: 3lbl [1/3 1/3 1/3] split
	M4: 3lbl [0.25 0.75 0.25] split: discard neutral to same size
	"""

	if not method in range(4+1):
		print 'Invalid method. Please choose M0, M1, M2, M3, M4'
		return

	#read in data
	print "Reading data..."
	training   = read_data("../../datasets/preprocessed/trainset.csv")
	test       = read_data("../../datasets/preprocessed/testset.csv")

	#set to default (no data is discarded)
	train_ind=[]


	print 'Processing labeling according to method M{0}'.format(method)

	if method==0:
		y,r = getLabels(training,test,splitProportions=[0.5,0.5])
	elif method ==1:
		y,r,labelEncoder = getLabels(training,test,splitProportions=[0.25,0.25],returnEncoder=True)

		train_ind = [n for n,yi in enumerate(y) if not yi==labelEncoder.transform('neutral') ] 
		test_ind  = [n for n,ri in enumerate(r) if not ri==labelEncoder.transform('neutral') ] 

		
	elif method ==2:
		y,r = getLabels(training,test,splitProportions=[0.25,0.25])
	elif method ==3:
		y,r = getLabels(training,test,splitProportions=[1.0/3,1.0/3])
	elif method ==4:
		y,r, labelEncoder = getLabels(training,test,splitProportions=[0.25,0.25],returnEncoder=True)
		
		#get indices for all classes for train and test
		train_i = [n for n,yi in enumerate(y) if yi==labelEncoder.transform('impolite') ] 
		train_n = [n for n,yi in enumerate(y) if yi==labelEncoder.transform('neutral') ] 
		train_p = [n for n,yi in enumerate(y) if yi==labelEncoder.transform('polite') ] 
		test_i  = [n for n,ri in enumerate(r) if ri==labelEncoder.transform('impolite') ] 
		test_n  = [n for n,ri in enumerate(r) if ri==labelEncoder.transform('neutral') ] 
		test_p  = [n for n,ri in enumerate(r) if ri==labelEncoder.transform('polite') ] 

		# Resize neutral class to same size other classes
		train_n = train_n[0:len(train_p)]
		test_n  =  test_n[0:len(test_p)]

		# fuse together and keep original order
		train_ind = sorted(train_i + train_n + train_p)
		test_ind  = sorted(test_i  + test_n  + test_p )

	if train_ind:
		#We only want to use the given indices
		print 'Discarding irrelevant data . . .'

		training  = [training[i] for i in train_ind]
		y         = [       y[i] for i in train_ind]
		test      = [    test[i] for i in test_ind]
		r         = [       r[i] for i in test_ind]

	if returnIndices:
		return training, test, y,r,train_ind, test_ind	
	else:
		return training, test, y,r	


if __name__ == '__main__':
	##	Main function. Demonstrates the label distributions under the 5 methods

	for i in range(5):
		print 'Method {0}'.format(i)
		_,_,y,r = getProcessedData(method=i)
		print 'Train'
		print_lbl_dist(y)
		print 'Test'
		print_lbl_dist(r)
