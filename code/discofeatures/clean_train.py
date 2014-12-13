#!/usr/bin/env python2
# coding=utf-8
import gc
import numpy as np
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes, decomposition
import sklearn
from featureDeduction import FeatureDeduction
from datapoint import *

def feature2vector(train_data,test_data,feature_deduction=None):
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

	return X, Xtest

def getLabeledData(training_data, test_data,splitPoints=[],splitProportions=[0.25, 0.25]):
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

	#print "Split points:"
	#print splitPoints

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

	return y,r

def print_lbl_dist(y):
	for l in set(y):
		print "{0}:{1}".format(l, 1.0 * sum([1 for p in y if p==0])/len(y))

def main():
	"""
	Main function. Runs and tests naive bayes on the data
	"""
	print "Reading training data."
	training = read_data("../../datasets/preprocessed/trainset.csv")
	test     = read_data("../../datasets/preprocessed/testset.csv")


	X, Xtest = feature2vector(training,test,1000)
	
	print "Setting up target"

	y,r = getLabeledData(training,test,splitProportions=[0.25,0.25])
	print 'Train'
	print_lbl_dist(y)
	print 'Test'
	print_lbl_dist(r)
	y,r = getLabeledData(training,test,splitProportions=[0.5,0.5])
	print 'Train'
	print_lbl_dist(y)
	print 'Test'
	print_lbl_dist(r)
	y,r = getLabeledData(training,test,splitProportions=[1.0/3,1.0/3])
	print 'Train'
	print_lbl_dist(y)
	print 'Test'
	print_lbl_dist(r)

	return

	print "Fitting classifier"

	classifier = naive_bayes.GaussianNB()
	classifier.fit(X, y)

	print "Fit classifier, calculating scores"

	print "Accuracy on test set:    ", classifier.score(Xtest,r)
	print "Accuracy on training set:", classifier.score(X,y)


if __name__ == '__main__':
	
	main()

