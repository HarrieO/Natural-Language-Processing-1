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
	#word_entropy = [ (0,float(score)) for score in post.read_column(2,'../../datasets/preprocessed/informationGainWords.txt') if not score=='']
	word_entropy = [ (0,float(score)) for score in post.read_column(1,'../../datasets/preprocessed/word_entropy.csv')    if not score=='']
	DOPf_entropy = [ (1,float(score)) for score in post.read_column(1,'../../datasets/preprocessed/informationGain.txt') if not score=='']


	feature_list  = sorted(word_entropy + DOPf_entropy,key=lambda tup: tup[1])
	feature_types = [tup[0] for tup in feature_list]


	num_word_features = feature_types[0:num_total_features].count(0)
	num_DOP_features  = feature_types[0:num_total_features].count(1)

	return num_word_features, num_DOP_features

def getBothFeatureData(num_word_features,num_DOP_features,method=2):
	"""
	Returns the data as a vector with wordcount and DOP features as well as labels
	"""

	#Load training, post data, labels and indices of datapoints that we use
	training, test, y,r,train_ind, test_ind = getProcessedData(method=method,returnIndices=True)

	if num_word_features > 0:
		# get word features
		print ">>> Getting {0} Word-count Features".format(num_word_features)
		X_w, Xtest_w = baseline.getTrainingTestFeatures(num_word_features, train_ind, test_ind)

	if num_DOP_features > 0:
		# get dop features
		print ">>> Getting {0} DOP-count featurs".format(num_DOP_features)
		X_d, Xtest_d = feature2vector(training,test,num_DOP_features)

	#Determine which features to return
	if num_word_features == 0:
		X 	  = X_d
		Xtest = Xtest_d
	elif num_DOP_features == 0:
		X     = X_w
		Xtest = Xtest_w
	else:
		# Fuse vectors
		print ">>> Fusing features to {0}".format(num_DOP_features+num_word_features)
		X     = np.concatenate((X_w,X_d),axis=1)
		Xtest = np.concatenate((Xtest_w,Xtest_d),axis=1)

	return X, Xtest, y, r



if __name__ == '__main__':


	################## MAIN CODE ############

	# method = 4
	# num_total_features = 1000

	# X, Xtest, y, r = getBothFeatureData( *getNumBothFeatures(num_total_features) , method=method)

	# print np.shape(X)
	# print np.shape(y)

	# print np.shape(Xtest)
	# print np.shape(r)

	# Train classifers etc.



	#########    DEBUG  DOP/WORDS Entropy   ###########

	word_entropy = [ (0,float(score)) for score in post.read_column(1,'../../datasets/preprocessed/word_entropy.csv') if not score=='']
	DOPf_entropy = [ (1,float(score)) for score in post.read_column(1,'../../datasets/preprocessed/informationGain.txt') if not score=='']


	feature_list  = sorted(word_entropy + DOPf_entropy,key=lambda tup: tup[1])
	feature_types = [tup[0] for tup in feature_list]

	features_list    = range(1, len(word_entropy)+len(DOPf_entropy)-1,1000)
	num_wordf_list   = [      feature_types[0:features].count(0) 		  for features in features_list]
	num_DOPf_list    = [	  feature_types[0:features].count(1)  		  for features in features_list]
	word_proportions = [1.0 * feature_types[0:features].count(0)/features for features in features_list]


	print "Word features: {0}".format(len(word_entropy))
	print "DOP  features: {0}".format(len(DOPf_entropy))


	plt.plot(features_list,num_wordf_list,'r',label='# Word features')
	plt.plot(features_list,num_DOPf_list,'g',label='#DOP features')
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel('total # of features')
	plt.ylabel('# of features')
	plt.legend()
	plt.show()


	plt.plot(features_list,word_proportions)
	plt.xscale('log')
	plt.xlabel('total # of features')
	plt.ylabel('Proportion of word features')
	plt.show()

	for features in [1,2,3,4,10, 1000,4000,10000,20000,200000]:
		
		print "Top {0} features: {1} words. {2} dop.".format(features, *getNumBothFeatures(features))
