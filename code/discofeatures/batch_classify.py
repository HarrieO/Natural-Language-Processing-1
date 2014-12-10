from sklearn import *
from clean_train import *
import time, re, amueller_mlp
from sklearn.metrics import *

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

def sort_results_csv(input_file='../../results/classifier_results.csv',output_file=''):
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
			[fd.write(settings_to_string(tup[0],tup[1],tup[2],tup[3],tup[4],tup[5],tup[6],tup[7],tup[8]) + "\n") for tup in table]


def settings_to_string(classifier_name,train_accuracy,test_accuracy,fit_time,score_time,
						features,classifier_settings='',train_conf_matrix='', test_conf_matrix=''):
	"""
	Get a string to store to csv file (also usefull for regexp)
	"""

	#add quotation marks for the strings, if needed
	if classifier_name==""     or not classifier_name[0]=="'":     classifier_name     = "'" + classifier_name	    + "'"
	if classifier_settings=="" or not classifier_settings[0]=="'": classifier_settings = "'" + classifier_settings	+ "'"
	
	return "{0},{1},{2},{3},{4},{5},{6},{7},{8}".format(classifier_name, train_accuracy,
				test_accuracy,fit_time,score_time,features, classifier_settings, 
				train_conf_matrix, test_conf_matrix)


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
	training   = read_data("../../datasets/preprocessed/trainset.csv")
	test       = read_data("../../datasets/preprocessed/testset.csv")
	y,r 	   = getLabels(training,test)


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
		regexp = settings_to_string(classifier_name,".*",".*",".*",".*",features,classifier_settings,".*",".*")
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

			#store results
			fd.write(settings_to_string(classifier_name,train_accuracy,
				test_accuracy,fit_time,score_time,features,classifier_settings,
				train_conf_matrix, test_conf_matrix) + "\n")

		#save to csv file and sort csv file
		fd.close()
		sort_results_csv()

def main():

	#tuples of classifers to test, and a string with their settings (to store)
	classifiers=[ (svm.SVC(kernel='poly'), 'kernel=poly'),
				  (svm.SVC(kernel='linear'), 'kernel=linear'),
				  #(gaussian_process.GaussianProcess(),''),
				  #(linear_model.LinearRegression(),''),
				  #(linear_model.Ridge(),''),
				  #(linear_model.Lasso(),''),
				  (naive_bayes.GaussianNB(),''),
				  (naive_bayes.MultinomialNB(),''),
				  (naive_bayes.BernoulliNB(),''),
				  (tree.DecisionTreeClassifier(),''),
				  (ensemble.RandomForestClassifier(),''),
				  (neighbors.nearest_centroid.NearestCentroid(),''),
				  #(sklearn.ensemble.GradientBoostingClassifier(),''),
				  (amueller_mlp.MLPClassifier(),''),
				  (sklearn.ensemble.AdaBoostClassifier(),''),
				  (sklearn.linear_model.Perceptron(n_iter=50),'n_iter=50'),
				  (svm.SVC(),'')
				 	]
		]

	# Maximum number of features: 261396
	features_set = logRange(261396,3,1)

	#combine combinatorial (factory because we dont want to duplicate all the classifiers)
	settings = ( (classifier, features, classifier_settings) for classifier,classifier_settings in classifiers for features in features_set)

	batch_run(settings)



if __name__ == '__main__':

	main()