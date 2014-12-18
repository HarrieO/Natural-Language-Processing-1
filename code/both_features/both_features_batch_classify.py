from both_features import *
import os, sys, csv
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
from batch_classify import *
from sklearn import *
from clean_train import *
import time, re, amueller_mlp
from sklearn.metrics import *

def both_features_batch_run(test_settings,method=2):
	"""
	batch_runs classifiers and stores results in the file ../../results/M{method}_classivier_results.csv
	Please provide settings as a tuple list:
	0:classifier object
	1:number of features (left after feature deduction)
	"""

	resultsfile = '../../results/M{0}_combined_features_results.csv'.format(method)


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
				X, Xtest, y, r = getBothFeatureData( *getNumBothFeatures(features) , method=method)
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




if __name__ == '__main__':

	#tuples of classifers to test, and a string with their settings (to store)
	classifiers=[ amueller_mlp.MLPClassifier(n_hidden=200),
				  amueller_mlp.MLPClassifier(n_hidden=400),
				  amueller_mlp.MLPClassifier(n_hidden=800),
				  ensemble.RandomForestClassifier(),
				  #sklearn.ensemble.AdaBoostClassifier(),
				  sklearn.linear_model.Perceptron(n_iter=60),
				  #svm.SVC(kernel='poly'),
				  #svm.SVC(kernel='linear'),
				  #svm.SVC(kernel='sigmoid'),
				  #naive_bayes.GaussianNB(),
				  #neighbors.nearest_centroid.NearestCentroid(),
				  #svm.SVC(),
				  tree.DecisionTreeClassifier(),
				  #naive_bayes.MultinomialNB(),
				  #naive_bayes.BernoulliNB(),
				  #sklearn.ensemble.GradientBoostingClassifier(),
				  #sklearn.ensemble.AdaBoostClassifier()
				 	]

	# Maximum number of features: 261396
	features_set = logRange(261396,15,1)

	#combine combinatorial (factory because we dont want to duplicate all the classifiers)
	settings = ( (classifier, features) for features in features_set for classifier  in classifiers )

	both_features_batch_run(settings,method=4)
	both_features_batch_run(settings,method=2)
	both_features_batch_run(settings,method=1)
	both_features_batch_run(settings,method=3)
	both_features_batch_run(settings,method=0)
	


