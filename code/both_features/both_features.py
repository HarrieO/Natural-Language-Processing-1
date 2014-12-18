import os, sys, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
from datapoint import *
from clean_train import *
import baseline
import post
import numpy as np
import matplotlib.pyplot as plt


def getNumBothFeatures(num_total_features):
	"""
	Returns how many dop and how many word features should be used, given a total number of features
	"""

	#load in the entropy for dop and word features
	word_entropy = [ (0,float(score)) for score in post.read_column(1,'word_entropy.csv') if not score=='']
	#word_entropy = [ (0,float(score)) for score in post.read_column(2,'../../datasets/preprocessed/informationGainWords.txt') if not score=='']
	DOPf_entropy = [ (1,-float(score)) for score in post.read_column(1,'../../datasets/preprocessed/informationGain.txt') if not score=='']


	feature_list  = sorted(word_entropy + DOPf_entropy,key=lambda tup: tup[1])
	feature_types = [tup[0] for tup in feature_list]


	num_word_features = feature_types[0:num_total_features].count(0)
	num_DOP_features  = feature_types[0:num_total_features].count(1)

	return num_DOP_features, num_word_features




def getIndexAndLabels(method=2):
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
	train_ind = range(len(training))
	test_ind  = range(len(test))


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

	return train_ind, test_ind, training, test, y,r	











################## MAIN CODE ############

for features in [1000,4000,10000,20000,200000]:
	
	print "Top {0} features: {1} words. {2} dop.".format(features, *getNumBothFeatures(features))



method = 4
num_DOP_features = 5
num_word_features = 4 

train_ind, test_ind, training, test, y,r	= getIndexAndLabels(method)

print len(train_ind)
print len(training)
print len(y)

#get dop features
X_d, Xtest_d = feature2vector(training,test,num_DOP_features)

#get words
X_w, Xtest_w = baseline.getTrainingTestFeatures(num_word_features, train_ind, test_ind)

print X_d.size
print X_w.size

print X_d
print X_w
