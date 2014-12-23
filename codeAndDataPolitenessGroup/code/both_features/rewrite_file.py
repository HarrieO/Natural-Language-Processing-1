from  both_features import *
from batch_classify import *

input_file = '../../results/M4_combined_features_results.csv'
output_file = 'tests.csv'

c_id = "DecisionTreeClassifier(compute_importances=None; criterion='gini'; max_depth=None; max_features=None; max_leaf_nodes=None; min_density=None; min_samples_leaf=1; min_samples_split=2; random_state=None; splitter='best')"


#print findRun(c_id,resultsfile=input_file, features=11)
#print findRun(c_id,resultsfile=input_file, word_features=9458,dop_features=12734)
with open(input_file, 'r') as f:
		header = f.readline()

word_entropy = [ (0,float(score)) for score in post.read_column(1,'../../datasets/preprocessed/word_entropy.csv')    if not score=='']
DOPf_entropy = [ (1,float(score)) for score in post.read_column(1,'../../datasets/preprocessed/informationGain.txt') if not score=='']

feature_list  = sorted(word_entropy + DOPf_entropy,key=lambda tup: tup[1])
feature_types = [tup[0] for tup in feature_list]



#load csv into table (automatically with correct datatypes)
table = np.recfromcsv(input_file,delimiter=',')
print 'word_features' in table.dtype.names
table.sort(order=['features','classifier_id'])
#table = sorted(table, key=lambda tup: tup['features'])

#table = sorted(table, key=lambda tup: tup['classifier_id'])

#print table


#with open(output_file,'w') as fd:
#			fd.write(header)
#			[fd.write(settings_to_string(tup[0],tup[1],tup[2],tup[3],tup[4],tup[5],tup[6],tup[7],feature_types[0:tup['features']].count(0),feature_types[0:tup['features']].count(1))+"\n") for tup in table]