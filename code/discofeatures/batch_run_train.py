from clean_train import *
import time

# Maximum number of features: 261396
feature_values = range(3); #range(50) + range(50,500,50) + range(500,5000,500) + range(5000,50000,5000) + range(50000,261396,50000)

#load to csv file to append the results
fd = open('classifier_results.csv','a')

#read in data
print "Reading training data."
training = read_data("trainset.csv")
test     = read_data("testset.csv")
y,r = getLabels(training,test)


for feature_value in feature_values:
	
	#do feature deduction
	X, Xtest = feature2vector(training,test,feature_value)

	#construct classifier
	classifier_name = 'naive_bayes';
	classifier_settings = 'name=gaussianNB'
	classifier = naive_bayes.GaussianNB()

	#fit classifier
	print "Fitting classifier"
	t0 = time.time()
	classifier.fit(X, y)
	fit_time = time.time() - t0

	print "Fit classifier, calculating scores"
	t0 = time.time()	
	test_accuracy = classifier.score(Xtest,r)
	train_accuracy = xlassifier.score(X,y)
	score_time = time.time()- t0

	#store results
	fd.write(classifer_name + ",{0},{1},{2},{3},{4}".format(train_accuracy,test_accuracy,fit_time,score_time,feature_value) + classifier_settings)

	#reset classifier
	classifier = None

fd.close()