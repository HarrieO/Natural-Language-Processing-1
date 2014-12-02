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

def getLabels(training_data, test_data):

	print "Setting up target"

	def giveLabel(score):
	    if post.score > 0.5:
	        return 'polite'
	    elif post.score > -0.5:
	        return 'neutral'
	    else:
	        return 'impolite'

	target = []
	for post in training_data:
	    target.append(giveLabel(post.score))
	real = []
	for post in test_data:
	    real.append(giveLabel(post.score))

	training_data = None
	test_data = None

	labelEncoder = preprocessing.LabelEncoder()

	# train targets
	y = labelEncoder.fit_transform(target)
	# true values of test data
	r = labelEncoder.transform(real)

	return y,r



def main():
	"""
	Main function. Runs and tests naive bayes on the data
	"""
	print "Reading training data."
	training = read_data("trainset.csv")
	test     = read_data("testset.csv")


	X, Xtest = feature2vector(training,test,1000)
	y,r = getLabels(training,test)

	print "Fitting classifier"

	classifier = naive_bayes.GaussianNB()
	classifier.fit(X, y)

	print "Fit classifier, calculating scores"

	# correct = 0
	# total   = 0
	# for x,t in zip(X,y):
	#     p = classifier.predict(x)
	#     if p == t:
	#         correct += 1
	#         print p, t
	#     total += 1.0
	#     print "Accuracy", (correct/total), "with", correct, "out of", total


	print "Accuracy on test set:    ", classifier.score(Xtest,r)
	print "Accuracy on training set:", classifier.score(X,y)

if __name__ == '__main__':
	main()

