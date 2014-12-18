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

def getBothFeatureData(num_word_features,num_DOP_features,method=2):
	"""
	Returns the data as a vector with wordcount and DOP features as well as labels
	"""

	#Load training, post data, labels and indices of datapoints that we use
	training, test, y,r,train_ind, test_ind = getProcessedData(method=method,returnIndices=True)

	# get dop features
	print ">>> Getting Word-count Features"
	X_w, Xtest_w = baseline.getTrainingTestFeatures(num_word_features, train_ind, test_ind)

	# get words
	print ">>> Getting DOP-count featurs"
	X_d, Xtest_d = feature2vector(training,test,num_DOP_features)

	# Fuse vectors
	print ">>> Fusing features"
	X     = np.concatenate((X_w,X_d),axis=1)
	Xtest = np.concatenate((Xtest_w,Xtest_d),axis=1)

	return X, Xtest, y, r



if __name__ == '__main__':


	################## MAIN CODE ############

	#for features in [1000,4000,10000,20000,200000]:
		
	#	print "Top {0} features: {1} words. {2} dop.".format(features, *getNumBothFeatures(features))


	method = 4
	num_total_features = 1000

	X, Xtest, y, r = getBothFeatureData( *getNumBothFeatures(num_total_features) , method=method)
	#num_DOP_features = 5
	#num_word_features = 4  
	#X, Xtest, y, r = getBothFeatureData(num_word_features,num_DOP_features,method)

	print np.shape(X)
	print np.shape(y)

	print np.shape(Xtest)
	print np.shape(r)


	#########    DEBUG ###########
	word_entropy = [ (0,float(score)) for score in post.read_column(1,'word_entropy.csv') if not score=='']
	DOPf_entropy = [ (1,-float(score)) for score in post.read_column(1,'../../datasets/preprocessed/informationGain.txt') if not score=='']


	feature_list  = sorted(word_entropy + DOPf_entropy,key=lambda tup: tup[1])
	feature_types = [tup[0] for tup in feature_list]

	features_list = range(1, len(word_entropy)+len(DOPf_entropy),1000)
	word_proportions = [1.0 * feature_types[0:features].count(0)/features for features in features_list]
	print len(features_list)
	print len(word_proportions)
	plt.plot(features,word_proportions)
	plt.show()
