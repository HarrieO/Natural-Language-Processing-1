from clean_train import *
import time, re, amueller_mlp
from sklearn import *

def logRange(limit, n=10,start_at_one=[]):
	"""
	returns an array of logaritmicly spaced integers untill limit of size n
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

def sort_results_csv(input_file='classifier_results.csv',output_file='classifier_results.csv'):
	"""
	Sorts the results file on featues (6th column) and classifier name (1th column) and stores results
	"""

	#import header first
	with open(input_file, 'r') as f:
		header = f.readline()

	table = np.recfromcsv(input_file,delimiter=',')
	#sort on features
	table = sorted(table, key=lambda tup: tup[5])
	#sort on classifier
	table = sorted(table, key=lambda tup: tup[0])

	#store sorted file
	with open(output_file,'w') as fd:
		fd.write(header + "\n")
		[fd.write(settings_to_string(tup[0],tup[1],tup[2],tup[3],tup[4],tup[5],tup[6]) + "\n") for tup in table]


def settings_to_string(classifier_name,train_accuracy,test_accuracy,fit_time,score_time,features,classifier_settings=''):
	return "'" + classifier_name + "',{0},{1},{2},{3},{4},'".format(train_accuracy,
				test_accuracy,fit_time,score_time,features) + classifier_settings + "'"


def batch_run(test_settings):
	"""
	batch_runs classifiers and stores results in the file ../../results/classivier_results.csv
	Please provide settings as a tuple list:
	0:classifier object
	1:number of features (left after feature deduction)
	2:a string explaining aditional settings that you adjusted in the classifier
	"""

	#read in data
	print "Reading data."
	training = read_data("../../datasets/preprocessed/trainset.csv")
	test     = read_data("../../datasets/preprocessed/testset.csv")
	y,r = getLabels(training,test)


	#initialize
	last_features = [];

	for settings in test_settings:

		#load to csv file to append the results. Do this in the loop to update the file live
		fd = open('../../results/classifier_results.csv','r+')

		#import parameters
		classifier 			= settings[0]
		features 			= settings[1]
		classifier_settings = settings[2]
		classifier_name 	= re.search(r".*'(.+)'.*", str(type(classifier))).groups()[0]
		
		#check if a experiment with the current settings was allready conducted (also move pointer to end of file)
		regexp = settings_to_string(classifier_name,".*",".*",".*",".*",features,classifier_settings)
		if len([1 for line in fd if re.search(regexp, line) != None]) > 0:
			print "Experiment with current settings was allready conducted, skipping"

		else:

			#do feature deduction if nesececary
			if not last_features == features: 
				X, Xtest = feature2vector(training,test,features)
				last_features = features

			#fit classifier
			print "Fitting " + classifier_name
			t0 = time.time()
			classifier.fit(X, y)
			fit_time = time.time() - t0

			#calculate scores
			print "Fit classifier, calculating scores"
			t0 = time.time()	
			test_accuracy = classifier.score(Xtest,r)
			train_accuracy = classifier.score(X,y)
			score_time = time.time()- t0

			#store results
			fd.write(settings_to_string(classifier_name,train_accuracy,
				test_accuracy,fit_time,score_time,features,classifier_settings) + "\n")

		#save to csv file and sort csv file
		fd.close()
		sort_results_csv()
		
def main():

	#classifiers to test:
	classifiers=[#gaussian_process.GaussianProcess(),
				 #linear_model.LinearRegression(),
				 #linear_model.Ridge(),
				 #linear_model.Lasso(),
				 naive_bayes.GaussianNB(),
				 naive_bayes.MultinomialNB(),
				 naive_bayes.BernoulliNB(),
				 #svm.SVC(),
				 #tree.DecisionTreeClassifier(),
				 #ensemble.RandomForestClassifier(),
				 #neighbors.nearest_centroid.NearestCentroid(),
				 #sklearn.ensemble.GradientBoostingClassifier(),
				 #sklearn.linear_model.Perceptron(),
				 #amueller_mlp.MLPClassifier(),
				 #sklearn.ensemble.AdaBoostClassifier()
				 	]


	# Maximum number of features: 261396
	features_set = logRange(261396,15,1)

	#in this case, settings are empty for all classifiers
	classifier_settings = '';

	#combine combinatorial (factory because we dont want to duplicate all the classifiers)
	settings = ( (classifier, classifier_settings, '') for features in features_set for classifier in classifiers)

	#run
	batch_run(settings)



if __name__ == '__main__':

	main()