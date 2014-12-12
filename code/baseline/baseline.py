import os, sys
import time, re
import numpy as np
import cPickle as pickle
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))
import post
import extractFeatures, amueller_mlp
from decisionstumps import *
from treedataToJoosttrees import getPostsWithTrees
from sklearn.metrics import *
import sklearn
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation, ensemble, svm, naive_bayes, decomposition, neighbors

'''
Settings
'''
forceExtractFeautres = True # If we want to extract features or not, otherwise load a Pickle file with preprocessed files, if the files are found
usePosTags = True
resultsPath = '../../results/baseline'+("_improved" if usePosTags else "") +'_classifier_results.csv'
maxFeatures = 10941 # 9479 words in training set, 10941 wordtags in training set

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


def getFeatures(trees, ignoredFeatures, features):
	global usePosTags
	results = list()
	i = 0
	for tree in trees:
		wordTags = getWordTagsFromTree(tree)
		if usePosTags:
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


def getTrainingTestFeatures(features):
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
		trainFeatures = getFeatures(treesTrain,ignoredWordTags,counts.keys())

		treesTest = [" ".join(row.trees) for row in testData]
		testFeatures = getFeatures(treesTest,ignoredWordTags,counts.keys())

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


def findRun(classifier_id,features):
	"""
	returns the numer of lines where the classifier /features combination occured
	if it didn't occur, return empty
	when one of the two features isn't set
	"""
	global resultsPath
	table = np.recfromcsv(resultsPath,delimiter=',')
	
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


def batch_run(test_settings):
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
	y,r 	   = getLabels()


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
		if findRun(classifier_id,features):
			print "Experiment with current settings was allready conducted, skipping"

		else:

			#load to csv file to append the results. Do this in the loop to update the file live
			fd = open(resultsPath,'a')

			#do feature deduction if nesececary
			if not last_features == features: 
				X, Xtest = getTrainingTestFeatures(features)
				last_features = features

		

			#fit classifier
			print "Fitting " + classifier_id
			t0 = time.time()
			print y[:20]
			classifier.fit(X, y)
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
			sort_results_csv(resultsPath)
		

def main():
	global maxFeatures


	#tuples of classifers to test, and a string with their settings (to store)
	classifiers=[ amueller_mlp.MLPClassifier(n_hidden=200),
				  amueller_mlp.MLPClassifier(n_hidden=400),
				  amueller_mlp.MLPClassifier(n_hidden=800),
				  sklearn.ensemble.RandomForestClassifier(),
				  sklearn.ensemble.AdaBoostClassifier(),
				  sklearn.linear_model.Perceptron(n_iter=50),
				  svm.SVC(kernel='poly'),
				  svm.SVC(kernel='linear'),
				  sklearn.naive_bayes.GaussianNB(),
				  sklearn.neighbors.nearest_centroid.NearestCentroid(),
				  sklearn.svm.SVC(),
				  sklearn.tree.DecisionTreeClassifier(),
				  #sklearn.naive_bayes.MultinomialNB(),
				  #sklearn.naive_bayes.BernoulliNB(),
				  sklearn.ensemble.GradientBoostingClassifier(),
				  sklearn.ensemble.AdaBoostClassifier(),
				  # sklearn.svm.SVC(kernel='rbf',gamma=0.4,C=1000000,class_weight={'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960}),
				  # sklearn.svm.SVC(kernel='rbf',gamma=0.2,C=1000000,class_weight={'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960}),
				  # sklearn.svm.SVC(kernel='rbf',gamma=0.8,C=1000000000,class_weight={'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960}),
				  # sklearn.svm.SVC(kernel='rbf',gamma=0.2,C=1000000000,class_weight={'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960}),
				  # sklearn.svm.SVC(kernel='linear',gamma=0.2,C=1000000000,class_weight={'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960}),
				  # sklearn.svm.SVC(kernel='poly',gamma=0.2,C=1000000000,class_weight={'positive':(4960+1830+1973)/1973,'negative':(4960+1830+1973)/1830,'neutral':(4960+1830+1973)/4960})
				 	]

	# Maximum number of features: 261396
	features_set = logRange(maxFeatures,15,1)

	#combine combinatorial (factory because we dont want to duplicate all the classifiers)
	settings = ( (classifier, features) for features in features_set for classifier  in classifiers )

	batch_run(settings)



if __name__ == "__main__":


		main()