from clean_train import *
import time

# Maximum number of features: 261396
feature_values = np.delete(logRange(10000,16),0) #drop the first element because we don't want to test feature reduction to 0

#load to csv file to append the results
fd = open('classifier_results.csv','a')

#read in data
print "Reading data."
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
	train_accuracy = classifier.score(X,y)
	score_time = time.time()- t0

	#store results
	fd.write("\n'" + classifier_name + "',{0},{1},{2},{3},{4},'".format(train_accuracy,test_accuracy,fit_time,score_time,feature_value) + classifier_settings + "'")

	#reset classifier
	classifier = None

fd.close()