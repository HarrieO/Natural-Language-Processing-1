import os, sys
import time, re
import numpy as np
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))
import post
import datapoint
import extractFeatures, amueller_mlp
from decisionstumps import *
from treedataToJoosttrees import getPostsWithTrees
from sklearn.metrics import *
import sklearn
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes, decomposition, neighbors
from sklearn.externals import joblib

'''
Settings
'''
forceExtractFeautres = True # If we want to extract features or not, otherwise load a Pickle file with preprocessed files, if the files are found
usePosTags = False
resultsPath = '../../results/M{0}_baseline'+("_improved" if usePosTags else "") +'_results.csv'
if usePosTags:
	maxFeatures = 10941 # 9479 words in training set, 10941 wordtags in training set
else:
	maxFeatures = 9479

def logRange(limit, n=10,start_at_one=[]):
	"""
	returns an array of logaritmicly spaced integers untill limit of size n
	starts at 0 unless if start_at_one = True
	"""

	if start_at_one: n=n+1

	if n > limit: raise Exception("n>limit!")

	result = [1]
	if n>1:  # just a check to avoid ZeroDivisionError
		ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
	while len(result)<n:
		next_value = result[-1]*ratio
		if next_value - result[-1] >= 1:
			# safe zone. next_value will be a different integer
			result.append(next_value)
		else:
			# problem! same integer. we need to find next_value by artificially incrementing previous value
			result.append(result[-1]+1)
			# recalculate the ratio so that the remaining values will scale correctly
			ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
	# round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
	logRange = np.array(map(lambda x: round(x)-1, result), dtype=np.uint64)
	if start_at_one:
		return np.delete(logRange,0)
	else:
		return logRange


def getFeaturesForSentence(sentence, extractionModel, usePosTags):
	trainFeatures = getFeatures([sentence],extractionModel['ignoredWordTags'],extractionModel['countKeys'], usePosTags, False)
	return extractionModel['vectorizer'].transform(trainFeatures)

def getFeatures(trees, ignoredFeatures, features, usePosTags, loadedAsTree = True):
	results = list()
	i = 0
	for tree in trees:
		if loadedAsTree:
			wordTags = getWordTagsFromTree(tree)
			if usePosTags:
				wordTags = [wordTag[0] for wordTag in wordTags]
			else:
				wordTags = [wordTag[0].split(" ")[1][:-1] for wordTag in wordTags]
		else:
			wordTags = re.findall(r"[\w']+|[\W]",tree)
		# results.append(extractFeatures.extract_features_word(wordTags, ignoredFeatures, features))
		results.append(dict(extractFeatures.extract_features_word(wordTags, ignoredFeatures, features)))
		# print str(float(i)/float(len(trees)))
		# if (i % 1000) == 0:
		# 	print i
		i += 1
	return results


def getTrainingTestFeatures(features, train_ind, test_ind):
	global forceExtractFeautres, usePosTags
	# Data structures
	classes			= ['negative','neutral','positive']
	classCutOff		= [-0.5,0.5]
	classCount 		= emptyClassCount(classes)
	wordTagCount 	= dict()
	totalScores		= dict()
	fileTrainData 	= '../../datasets/preprocessed/baselineTrainSet'+str(features)+str(usePosTags)+'.p'
	fileTestData	= '../../datasets/preprocessed/baselineTestSet'+str(features)+str(usePosTags)+'.p'

	testData = getPostsWithTrees('../../datasets/preprocessed/')
	
	if os.path.isfile(fileTrainData) and os.path.isfile(fileTestData) and not forceExtractFeautres:
		# Skip file extraction if preprocessed files are available
		trainFeatures = pickle.load(open(fileTrainData,'rb'))
		testFeatures = pickle.load(open(fileTestData,'rb'))
	else:
		# First extract the counts
		counts, ignoredWordTags = reduceFeatureSpace(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'),classes,classCutOff,wordTagCount,classCount,totalScores, features, usePosTags)
		print len(counts.keys())
		print len(ignoredWordTags)
		treesTrain = post.read_column(4,os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/discotrain.csv'))
		trainFeatures = getFeatures(treesTrain,ignoredWordTags,counts.keys(), usePosTags)

		treesTest = [" ".join(row.trees) for row in testData]
		testFeatures = getFeatures(treesTest,ignoredWordTags,counts.keys(), usePosTags)
		if len(train_ind) > 0:
			trainFeatures = [trainFeatures[ind] for ind in train_ind]
			testFeatures = [testFeatures[ind] for ind in test_ind]
		print len(trainFeatures)
		# Store results in a pickle file
		pickle.dump(trainFeatures, open(fileTrainData,'w+b'))
		pickle.dump(testFeatures, open(fileTestData,'w+b'))


	# trainScores = [float(row) for row in post.read_column(1,'../../datasets/preprocessed/train.csv')]
	# trainClasses = scoresToClass(trainScores,classCutOff,classes)
	# counts = {'neutral':0,'positive':0,'negative':0}
	# maxCounts = trainClasses.count('negative')
	# i = 0
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

	vectorizer = feature_extraction.DictVectorizer(sparse=False)
	X     = vectorizer.fit_transform(trainFeatures)
	Xtest = vectorizer.transform(testFeatures)
	pickle.dump({'ignoredWordTags': ignoredWordTags, 'countKeys': counts.keys(), 'vectorizer': vectorizer}, open('../../results/models/last_extraction.p','w+b'))
	return X, Xtest

def getLabels():
	# Data structures
	classes			= ['negative','neutral','positive']
	classCutOff		= [-0.5,0.5]

	testData = getPostsWithTrees('../../datasets/preprocessed/')
	trainScores = [float(row) for row in post.read_column(1,'../../datasets/preprocessed/train.csv')]
	testScores = [row.score for row in testData]
	trainClasses = scoresToClass(trainScores,classCutOff,classes)
	testClasses = scoresToClass(testScores,classCutOff,classes)


	# counts = {'neutral':0,'positive':0,'negative':0}
	# maxCounts = trainClasses.count('negative')
	# i = 0
	# newTrainClasses = list()
	# while i<len(trainClasses):
	# 	if counts[trainClasses[i]]<maxCounts:
	# 		newTrainClasses.append(trainClasses[i])
	# 		counts[trainClasses[i]]+=1
	# 	i += 1
	# trainClasses = newTrainClasses

	return trainClasses, testClasses


def sort_results_csv(input_file='../../results/baseline_classifier_results.csv',output_file=''):
	"""
	Sorts the results csv file and writes to the same file.
	Sort on classifier name first (1th column), then on features (6th column)
	"""

	if output_file =='': output_file = input_file

	#import header first
	with open(input_file, 'r') as f:
		header = f.readline()

	#load csv into table (automatically with correct datatypes)
	table = np.recfromcsv(input_file,delimiter=',')

	#only sort if we have more then one element (to prevent bugs)
	if np.size(table) > 1:

		#sort on features
		table = sorted(table, key=lambda tup: tup[5])
		#sort on classifier
		table = sorted(table, key=lambda tup: tup[0])

		#store sorted file
		with open(output_file,'w') as fd:
			fd.write(header)
			[fd.write(settings_to_string(tup[0],tup[1],tup[2],tup[3],tup[4],tup[5],tup[6],tup[7]) + "\n") for tup in table]


def findRun(classifier_id,features,resultsfile = '../../results/classifier_results.csv'):
	"""
	returns the numer of lines where the classifier /features combination occured
	if it didn't occur, return empty
	when one of the two features isn't set
	"""

	table = np.recfromcsv(resultsfile,delimiter=',')
	
	#make sure table is allways iterable
	if np.size(table)==1: table=list(table.flatten())

	return [n for n,tup in enumerate(table) if tup[0]=='"' + classifier_id + '"' and tup[5]==features]




def settings_to_string(classifier_id,train_accuracy,test_accuracy,fit_time,score_time,
						features,train_conf_matrix='', test_conf_matrix=''):
	"""
	Get a string to store to csv file (also usefull for regexp)
	"""

	#add quotation marks for the strings, if needed
	if classifier_id==""     or not classifier_id[0]=='"':     classifier_id     = '"' + classifier_id	    + '"'
	
	return "{0},{1},{2},{3},{4},{5},{6},{7}".format(classifier_id, train_accuracy,
				test_accuracy,fit_time,score_time,features, 
				train_conf_matrix, test_conf_matrix)


def batch_run(test_settings,method=2):
	"""
	batch_runs classifiers and stores results in the file ../../results/classivier_results.csv
	Please provide settings as a tuple list:
	0:classifier object
	1:number of features (left after feature deduction)
	2:a string explaining aditional settings that you adjusted in the classifier
	"""
	global resultsPath
	classes			= ['negative','neutral','positive']
	#read in data
	print "Reading data."
	train_ind, test_ind, y,r = getIndicesAndLabels(method)

	#initialize
	last_features = [];

	for settings in test_settings:


		#import parameters
		classifier 			= settings[0]
		features 			= settings[1]
		#converte the class name and properties into a string, be carefull for punctuation in csv
		classifier_id = str(classifier)
		classifier_id = classifier_id.replace('\n', ' ').replace('"',"'").replace(',',';')
		classifier_id = ' '.join(classifier_id.split())

		#check if a experiment with the current settings was allready conducted
		if False and findRun(classifier_id,features,resultsfile=resultsPath[:].format(method)):
			print "Experiment with current settings was allready conducted, skipping"

		else:

			#load to csv file to append the results. Do this in the loop to update the file live
			fd = open(resultsPath[:].format(method),'a')

			#do feature deduction if nesececary
			if not last_features == features: 
				X, Xtest = getTrainingTestFeatures(features, train_ind, test_ind)
				# This is a hack TODO: Fix the hack
				last_features = features

		

			#fit classifier
			print "Fitting " + classifier_id
			t0 = time.time()
			print y[:20]
			classifier.fit(X, y)
			joblib.dump(classifier, '../../results/models/last_classifier.p') 
			fit_time = time.time() - t0

			#Predict labels
			print "Fit classifier, calculating scores"
			t0 = time.time()	
			y_pred = classifier.predict(X)
			r_pred = classifier.predict(Xtest)
			score_time = time.time()- t0

			#calculate performances
			train_accuracy  = accuracy_score(y,y_pred)
			test_accuracy   = accuracy_score(r,r_pred)
			train_conf_matrix = np.array_str(confusion_matrix(y,y_pred) ).replace("\n",' ')
			test_conf_matrix  = np.array_str(confusion_matrix(r,r_pred) ).replace("\n",' ')
			print "Training"
			print(classification_report(y,y_pred, target_names=classes))
			print "Testing"
			print(classification_report(r,r_pred, target_names=classes))

			#store results
			fd.write(settings_to_string(classifier_id,train_accuracy,
				test_accuracy,fit_time,score_time,features,
				train_conf_matrix, test_conf_matrix) + "\n")

			#save to csv file and sort csv file
			fd.close()
			sort_results_csv(resultsPath[:].format(method))
		


## This is an absolute hack
## TODO: Fix this to something reasonable
def getIndicesAndLabels(method=2):
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
	training   = datapoint.read_data("../../datasets/preprocessed/trainset.csv")
	test       = datapoint.read_data("../../datasets/preprocessed/testset.csv")

	#set to default (no data is discarded)
	train_ind=[]
	test_ind=[]

	print 'Processing labeling according to method M{0}'.format(method)
	if method==0:
		y,r = getLabelsFix(training,test,splitProportions=[0.5,0.5])
	elif method ==1:
		y,r,labelEncoder = getLabelsFix(training,test,splitProportions=[0.25,0.25],returnEncoder=True)

		train_ind = [n for n,yi in enumerate(y) if not yi==labelEncoder.transform('neutral') ] 
		test_ind  = [n for n,ri in enumerate(r) if not ri==labelEncoder.transform('neutral') ] 

		
	elif method ==2:
		y,r = getLabelsFix(training,test,splitProportions=[0.25,0.25])
	elif method ==3:
		y,r = getLabelsFix(training,test,splitProportions=[1.0/3,1.0/3])
	elif method ==4:
		y,r, labelEncoder = getLabelsFix(training,test,splitProportions=[0.25,0.25],returnEncoder=True)
		
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

	return train_ind, test_ind, y,r	

## This is another stupid hack
def getLabelsFix(training_data, test_data,splitPoints=[],splitProportions=[0.25, 0.25],verbose=True,returnEncoder=False):
	"""
	Returns labels for the data, splits data into proportions given by 
	splitProportions = [propNegative, propPositive]
	or absolute values given by splitPoints = [Negativepoint, Positivepoint]
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

def main():
	global maxFeatures


	#tuples of classifers to test, and a string with their settings (to store)
	classifiers=[ 
				  sklearn.svm.SVC()
				 	]

	# Maximum number of features: 261396
	features_set = logRange(261396,15,1)

	#combine combinatorial (factory because we dont want to duplicate all the classifiers)
	settings = ( (classifier, features) for features in features_set for classifier  in classifiers )

	batch_run(settings, 3)
	batch_run(settings, 4)



if __name__ == "__main__":


		main()