import numpy as np
import matplotlib.pyplot as plt
import textwrap
import sys, os
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

def print_conf_matrices(mat_list):

	if isinstance(mat_list[0],str): mat_list = [reproduce_conf_matrix(mat) for mat in mat_list]
	num_mats = len(mat_list)
	num_classes =  len(mat_list[0])

	if num_classes ==3:
		signs = ['+','0','-']

		print "   __Predicted___   " * num_mats
		print " _|__+_|__0_|__-_|  " * num_mats

		for n in range(3):
			for mat in mat_list:
				print "|{0}|{1: >4}|{2: >4}|{3: >4}| ".format(signs[n],mat[n][0],mat[n][1],mat[n][2]),
			print " "

	elif num_classes ==2:

		signs = ['+','-']

		print "   Predicted   " * num_mats
		print " _|__+_|__-_|  " * num_mats

		for n in range(2):
			for mat in mat_list:
				print "|{0}|{1: >4}|{2: >4}| ".format(signs[n],mat[n][0],mat[n][1]),
			print " "

def print_all_conf_mats(train_mat_list,test_mat_list,features_list):
	"""
	Prints a row of matrices that keeping track of terminal width
	"""
	num_classes =  len(train_mat_list[0])
	num_mats = len(train_mat_list)

	if num_classes ==2:
		mat_width = 15
	else:
		mat_width = 19
	max_mats = int(os.popen('stty size', 'r').read().split()[1])/mat_width
	rows =  int(np.ceil(1.0 * num_mats/max_mats))

	for r in range(rows):
		ind0 = r * max_mats
		ind1 = ind0 + max_mats

		print  "".join( ["Features:{0: >8}  |".format(f) for f in features_list[ind0:ind1] ])
		print "Training data:"
		print_conf_matrices(train_mat_list[ind0:ind1])
		print "Test data"
		print_conf_matrices(test_mat_list[ind0:ind1])
		print "\n\n\n"


def get_result_table(filename='classifier_results.csv'):
	"""
	Return recarray with results.
	"""

	table = np.recfromcsv(filename,delimiter=',')
	#remove quotation marks from 1th column
	for i in range(table.size):
		table[i]['classifier_id'] = table[i]['classifier_id'][1:-1]

	return table

def plot_classifier_results(c_id,plot_runtime=True,table=[],compareAcc=[],compareLbl='Baseline'):
	"""
	Plots the performance of current classifier of different features,
	Option to plot runtime as well
	"""

	if table==[]: table = get_result_table()

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

	plt.title("{0} ({1})".format(class_id_to_tuple(c_id)[0], class_id_list(table=c_table).index(c_id)))
	plt.show()

def class_id_to_tuple(c_id):
	"""
	Returns a tuple of (classifier name, classifier settings)
	"""
	return re.findall(r"(.*)\((.*)\)",c_id)[0]

def class_id_list(filename='classifier_results.csv',table=[]):
	if not table==[]:
		return list(set(table['classifier_id']))
	else:
		return list(set(post.read_column(0,filename)))

def main():

	#which data do we want to plot:
	print '           -   0   +'
	print 'M0: 2lbl [50%     50%]                    '
	print 'M1: 2lbl [25% 50% 25%] discard neutral    >>> [50%     50%]'
	print 'M2: 3lbl [25% 50% 25%]'
	print 'M3: 3lbl [33% 33% 33%]'
	print 'M4: 3lbl [25% 50% 25%] resize neutral     >>> [33% 33% 33%]'
	method = int(raw_input('Method:'))

	print '0: DOP features'
	print '1: baseline'
	print "2: baseline 'improved'"
	print "3: word+DOP features"

	feature_classifiers = ['classifier','baseline','baseline_improved','combined_features']
	f = int(raw_input('Type of features/classifier:'))
	
	#which data do we want to plot
	filename = 'M{0}_{1}_results.csv'.format(method,feature_classifiers[f])

	#all unique classifiers (and settings)
	classifier_id_list = class_id_list(filename=filename)

	#split into name and settings
	classifier_tuple_list = [class_id_to_tuple(c) for c in classifier_id_list] 

	print "==============================================="
	print "All classifiers:"
	print "==============================================="
	#print "\n".join([str(n)+". "+str(c) for n,c in enumerate(classifier_list)])
	print "\n".join(["{0: >2}. {1}\n {2}...".format(n, c[0],
		textwrap.fill(c[1], initial_indent='    > ', subsequent_indent='      ')) for n,c in enumerate(classifier_tuple_list)])
	print 
	#print "\n".join([str(n) + c[0] for n,c in enumerate(classifier_tuple_list])



	best_n = 10

	#get sorted table
	table = get_result_table(filename=filename)

	best_table = sorted(table, key=lambda row: row['test_accuracy'], reverse=True)[0:best_n]

	print "\n\n"
	print "==============================================="
	print "Best {0} classifiers:".format(best_n)
	print "==============================================="

	print "\n".join(["{0: >2}.{1: >27} ({2: >2}) with {3: >8} features. Train:{4:.4f}. Test{5:.4}.".format(n+1, class_id_to_tuple(row['classifier_id'])[0],
										classifier_id_list.index(row['classifier_id']),
										row['features'], row['train_accuracy'], row['test_accuracy'])
									for n,row in enumerate(best_table)])



	if raw_input('Plot graphs? (y/n):').lower()=='y':
		if raw_input('Plot all graphs? (y/n):').lower()=='y':
			[plot_classifier_results(c_id,table=table) for c_id in classifier_id_list]
		else:

			index = int(raw_input('Which classifier? (index):'))
			c_id = classifier_id_list[index];
			print "==============================================================================================================================="
			print "=== " + c_id + " ==="
			print "==============================================================================================================================="
		
			print_all_conf_mats(table[table['classifier_id']==c_id]['train_conf_matrix'],
						        table[table['classifier_id']==c_id]['test_conf_matrix'],
						        table[table['classifier_id']==c_id]['features'] )

			plot_classifier_results(c_id,table=table)

if __name__ == '__main__':
	main()


