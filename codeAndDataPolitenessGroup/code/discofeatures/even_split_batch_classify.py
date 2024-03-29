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
		table = sorted(table, key=lambda tup: tup['features'])
		#sort on classifier
		table = sorted(table, key=lambda tup: tup['classifier_id'])

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

def even_split_correction(training,y):

	print "Correcting split"

	ind0 = [n for n,yi in enumerate(y) if yi==0]
	ind1 = [n for n,yi in enumerate(y) if yi==1]
	ind2 = [n for n,yi in enumerate(y) if yi==2]

	maxN =  len(ind0) #hardcoded (this is the max)

	ind0 = ind0[0:maxN]
	ind1 = ind1[0:maxN]
	ind2 = ind2[0:maxN]

	use_ind = sorted(ind0 + ind1 + ind2)

	return [training[i] for i in use_ind], [y[i] for i in use_ind]

def batch_run(test_settings):
	"""
	batch_runs classifiers and stores results in the file ../../results/classivier_results.csv
	Please provide settings as a tuple list:
	0:classifier object
	1:number of features (left after feature deduction)
	"""

	resultsfile = '../../results/classifier_results_split_corrected.csv'

	#read in data
	print "Reading data."
	training   = read_data("../../datasets/preprocessed/trainset.csv")
	test       = read_data("../../datasets/preprocessed/testset.csv")
	y,r 	   = getLabels(training,test)

	training,y = even_split_correction(training,y)

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
		if findRun(classifier_id,features,resultsfile=resultsfile):
			print "Experiment with current settings was allready conducted, skipping"

		else:

			#load to csv file to append the results. Do this in the loop to update the file live
			fd = open(resultsfile,'a')

			#do feature deduction if nesececary
			if not last_features == features: 
				X, Xtest = feature2vector(training,test,features)
				last_features = features

			#fit classifier
			print "Fitting " + classifier_id
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
			fd.write(settings_to_string(classifier_id,train_accuracy,
				test_accuracy,fit_time,score_time,features,
				train_conf_matrix, test_conf_matrix) + "\n")

			#save to csv file and sort csv file
			fd.close()
			sort_results_csv(input_file=resultsfile)

def main():

	#tuples of classifers to test, and a string with their settings (to store)
	classifiers=[ amueller_mlp.MLPClassifier(n_hidden=200),
				  amueller_mlp.MLPClassifier(n_hidden=400),
				  amueller_mlp.MLPClassifier(n_hidden=800),
				  ensemble.RandomForestClassifier(),
				  sklearn.ensemble.AdaBoostClassifier(),
				  sklearn.linear_model.Perceptron(n_iter=50),
				  svm.SVC(kernel='poly'),
				  svm.SVC(kernel='linear'),
				  naive_bayes.GaussianNB(),
				  neighbors.nearest_centroid.NearestCentroid(),
				  svm.SVC(),
				  tree.DecisionTreeClassifier(),
				  #naive_bayes.MultinomialNB(),
				  #naive_bayes.BernoulliNB(),
				  sklearn.ensemble.GradientBoostingClassifier(),
				  sklearn.ensemble.AdaBoostClassifier()
				 	]

	# Maximum number of features: 261396
	features_set = logRange(261396,15,1)

	#combine combinatorial (factory because we dont want to duplicate all the classifiers)
	settings = ( (classifier, features) for features in features_set for classifier  in classifiers )

	batch_run(settings)



if __name__ == '__main__':

	main()