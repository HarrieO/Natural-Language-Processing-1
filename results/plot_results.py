import sys, os
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from operator import add, itemgetter

sys.path.append('../code')
sys.path.append('../code/discofeatures')

import post
from batch_classify import *

def reproduce_conf_matrix(mat_string):
	"""
	Reproduce actual matrix from string as stored in csv file
	"""
	#remove [[ and ]] at the beginning / end
	mat_string = mat_string[2:-2]

	#remove multiple whitespaces
	mat_string = ' '.join(mat_string.split())

	#split to 2d array
	mat = [row.split(' ') for row in mat_string.split('] [')]

	#return as int 
	return [[int(i) for i in row if not i==''] for row in mat]

def print_conf_matix(mat):
	if isinstance(mat,str): mat = reproduce_conf_matrix(mat)

	signs = ['+','0','-']
	print "   __Predicted___"
	print " _|__+_|__0_|__-_|"

	for n,row in enumerate(mat):
		print "{0} |{1: >4}|{2: >4}|{3: >4}|".format(signs[n],row[0],row[1],row[2])



def get_result_table(filename='classifier_results.csv'):
	"""
	Return recarray with results.
	"""

	table = np.recfromcsv('classifier_results.csv',delimiter=',')
	#remove quotation marks from 1th column
	for i in range(table.size):
		table[i]['classifier_id'] = table[i]['classifier_id'][1:-1]

	return table

def plot_classifier_results(c_id,plot_runtime=True,table=[],compareAcc=[],compareLbl='Baseline'):
	"""
	Plots the performance of current classifier of different features,
	Option to plot runtime as well
	"""

	if not table: table = get_result_table()

	c_table =  table[table["classifier_id"]==c_id]

	fig, ax1 = plt.subplots()
	ax1.plot(c_table['features'], c_table['train_accuracy'], 'b-s',label='Training data')
	ax1.plot(c_table['features'], c_table['test_accuracy'], 'r-s', label='Test data')
	ax1.set_xlabel('# of features')
	ax1.set_xscale('log')

	ax1.yaxis.grid()
	ax1.xaxis.grid()

	ax1.set_ylabel('Performance (proportion correct)')
	ax1.set_ylim([0,1])

	if compareAcc:
		ax1.plot(c_table['features'],[compareAcc for _ in c_table['features']],'-',color='orange',label=compareLbl)

	lines, labels = ax1.get_legend_handles_labels()
	
	if plot_runtime:
		ax2 = ax1.twinx()
		ax2.plot(c_table['features'], map(add,c_table['fit_time'],c_table['score_time']), 'g-8',label='Runtime')
		ax2.set_ylabel('Runtime (s)')
		lines2, labels2 = ax2.get_legend_handles_labels()

		lines.append(lines2[0])
		labels.append(labels2[0])

	ax1.legend(lines, labels)

	plt.title("{0} ({1})".format(class_id_to_rowle(c_id)[0], class_id_list().index(c_id)))
	plt.show()

def class_id_to_rowle(c_id):
	"""
	Returns a rowle of (classifier name, classifier settings)
	"""
	return re.findall(r"(.*)\((.*)\)",c_id)[0]

def class_id_list():
	return list(set(post.read_column(0,'classifier_results.csv')))

def main():

	#all unique classifiers (and settings)
	classifier_id_list = class_id_list()

	classifier_rowle_list = [class_id_to_rowle(c) for c in classifier_id_list] 

	print [c for c,_ in classifier_rowle_list]

	#return [x for x in set(post.read_column(0,'classifier_results.csv'))]


	print "==============================================="
	print "All classifiers:"
	print "==============================================="
	#print "\n".join([str(n)+". "+str(c) for n,c in enumerate(classifier_list)])
	print "\n".join(["{0: >2}. {1}\n {2}...".format(n, c[0],
		textwrap.fill(c[1], initial_indent='    > ', subsequent_indent='      ')) for n,c in enumerate(classifier_rowle_list)])
	print 
	#print "\n".join([str(n) + c[0] for n,c in enumerate(classifier_rowle_list])





	best_n = 10

	#get sorted table
	table = get_result_table()
	best_table = sorted(table, key=lambda row: row['test_accuracy'], reverse=True)[0:best_n]

	print "\n\n"
	print "==============================================="
	print "Best {0} classifiers:".format(best_n)
	print "==============================================="

	print "\n".join(["{0: >2}.{1: >27} ({2: >2}) with {3: >8} features. Train:{4:.4f}. Test{5:.4}.".format(n+1, class_id_to_rowle(row['classifier_id'])[0],
										classifier_id_list.index(row['classifier_id']),
										row['features'], row['train_accuracy'], row['test_accuracy'])
									for n,row in enumerate(best_table)])




	if True: #raw_input('Plot graphs? (y/n):').lower()=='y':
		if False: #raw_input('Plot all graphs? (y/n):').lower()=='y':
			[plot_classifier_results(c_id) for c_id in classifier_id_list]
		else:

			index = 3 #int(raw_input('Which classifier? (index):'))
			c_id = classifier_id_list[index];
			print "==============================================================================================================================="
			print "=== " + c_id + " ==="
			print "==============================================================================================================================="
			
			for row in table[table['classifier_id']==c_id]:
				print 'Features: ', row['features'], 'Train conf.matrix:'
				print_conf_matix(row['test_conf_matrix'])


			plot_classifier_results(c_id)

if __name__ == '__main__':
	main()


