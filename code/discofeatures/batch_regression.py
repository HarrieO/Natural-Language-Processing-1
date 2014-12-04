from clean_train import *
import time, re
from sklearn import *
from batch_classify import *


def batch_regression(features_list):
	"""
	batch_runs regression and stores results in the file ../../results/classivier_results.csv
	"""

	#read in data
	print "Reading data."
	training = read_data("../../datasets/preprocessed/trainset.csv")
	test     = read_data("../../datasets/preprocessed/testset.csv")
	y   = [post.score for post in training]
	r   = [post.score for post in test]


	#initialize
	last_features = [];

	for features in features_list:

		#load to csv file to append the results. Do this in the loop to update the file live
		fd = open('../../results/classifier_results.csv','r+')

		#import parameters
		classifier 			= linear_model.LinearRegression()
		classifier_name 	= re.search(r".*'(.+)'.*", str(type(classifier))).groups()[0]
		classifier_settings = "performance=MSE"  #for clarity
		
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
			rPred = classifier.predict(Xtest)
			yPred = classifier.predict(X)
			test_accuracy  = sklearn.metrics.mean_squared_error(rPred,r)
			train_accuracy = sklearn.metrics.mean_squared_error(yPred,y)
			score_time = time.time()- t0

			#store results
			fd.write("\n" + settings_to_string(classifier_name,train_accuracy,
				test_accuracy,fit_time,score_time,features,classifier_settings))

		fd.close()
		
def main():

	# Maximum number of features: 261396
	features_list = logRange(261396,15,1)


	#run
	batch_regression(features_list)


if __name__ == '__main__':

	main()