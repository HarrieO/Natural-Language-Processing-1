import sys, os
import numpy as np
import matplotlib.pyplot as plt
from operator import add

sys.path.append('../code')
sys.path.append('../code/discofeatures')
#sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))

import post
from batch_classify import *


def get_classifier_list():
	"""
	Returns the set of different classifiers that we have results of
	"""
	#get list with unique values for classifier (without the quotes at begin and end)
	return [x[1:-1] for x in set(post.read_column(0,'classifier_results.csv'))]

def get_classifier_table(classifier_name,sort_ind=5):
	"""
	Returns the results of the given classifier in a nested list.
	Each nested list corresponds to a variabele, the table is sorted according to index in the csv file 
	Default sort = column 5 (features)
	"""

	classifier_settings = '.*' #not implemented yet

	#get all lines with current classifier and current settings
	regexp = settings_to_string(classifier_name,".*",".*",".*",".*",".*",classifier_settings)
	with open('classifier_results.csv','r') as fd:
		lines = [line for line in fd if re.search(regexp, line) != None]

	table = [];

	for line in lines:
		v = line.split(',')
		#v[0] #classifier
		#v[1] #train_performance
		#v[2] #test_performance
		#v[3] #fit time
		#v[4] #score time
		#v[5] #features
		#v[6] #settings
		#v[7] #\n
		row = [float(v[1]), float(v[2]),float(v[3]),float(v[4]),int(v[5])]
		table.append(row)

	#sort on index minus one because we dropped the first index above
	table.sort(key=lambda x: x[sort_ind-1])

	#transpose table for easy indexing (each nested list is a variabele)
	table = [list(x) for x in zip(*table)]

	return table

def plot_classifier_results(classifier_name,plot_runtime=True):
	"""
	Plots the performance of current classifier of different features,
	Option to plot runtime as well
	"""

	table = get_classifier_table(classifier_name)

	#export for readability
	train_p  = table[0]
	test_p   = table[1]
	fit_t    = table[2]
	score_t  = table[3]
	run_t    = map(add, fit_t, score_t)
	features = table[4]

	fig, ax1 = plt.subplots()
	ax1.plot(features, train_p, 'b-s',label='Training data')
	ax1.plot(features, test_p, 'r-s', label='Test data')
	ax1.set_xlabel('# of features')

	ax1.set_ylabel('Performance (proportion correct)')

	lines, labels = ax1.get_legend_handles_labels()
	
	if plot_runtime:
		ax2 = ax1.twinx()
		ax2.plot(features, run_t, 'g-8',label='Runtime')
		ax2.set_ylabel('Runtime (s)')
		lines2, labels2 = ax2.get_legend_handles_labels()

		lines.append(lines2[0])
		labels.append(labels2[0])

	ax1.legend(lines, labels)

	plt.title(classifier_name)
	plt.grid()
	plt.show()

def main():

	classifier_list = get_classifier_list()

	print classifier_list

	[plot_classifier_results(classifier_name) for classifier_name in classifier_list]



if __name__ == '__main__':
	main()


