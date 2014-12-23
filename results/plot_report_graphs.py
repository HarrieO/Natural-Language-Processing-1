from plot_results import *

#settings!
proportional_features = True  #absolute or proportional features on x axis
plot_apart			  = True #plot word and dop in same or different plot

#load results
w_table = get_result_table(filename='M4_baseline_results.csv')
d_table = get_result_table(filename='M4_classifier_results.csv')
b_table = get_result_table(filename='M4_combined_features_results.csv')

#Dop features: 261394
# word featuers:9477
#Hack
w_table =w_table[w_table['features']<=9477]

#classifiers to plot
c_id_list = ["MLPClassifier(batch_size=100; l2decay=0; loss='cross_entropy'; lr=0.1; n_hidden=800; output_layer=None; verbose=0)",
			 "SVC(C=1.0; cache_size=200; class_weight=None; coef0=0.0; degree=3; gamma=0.0; kernel='linear'; max_iter=-1; probability=False; random_state=None; shrinking=True; tol=0.001; verbose=False)",
			 "SVC(C=1.0; cache_size=200; class_weight=None; coef0=0.0; degree=3; gamma=0.0; kernel='rbf'; max_iter=-1; probability=False; random_state=None; shrinking=True; tol=0.001; verbose=False)",
			 "Perceptron(alpha=0.0001; class_weight=None; eta0=1.0; fit_intercept=True; n_iter=60; n_jobs=1; penalty=None; random_state=0; shuffle=False; verbose=0; warm_start=False)",
			 "DecisionTreeClassifier(compute_importances=None; criterion='gini'; max_depth=None; max_features=None; max_leaf_nodes=None; min_density=None; min_samples_leaf=1; min_samples_split=2; random_state=None; splitter='best')"]

c_names_list = ['MLP','linear SVM','rbf SVM', 'Perceptron','Decision Tree']

color_list = ['purple','green','turquoise','red','blue','gray']

if proportional_features:
	s0 = 1.0/9477
	s1 = 1.0/261394 
else:
	s0=s1=1.0

if plot_apart:
	f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
else:
	f, ax1 = plt.subplots()
	ax2 = ax1

for n,c_id in enumerate(c_id_list):

	c_table = w_table[w_table['classifier_id']==c_id]

	print c_table['features']
	print c_table['test_accuracy']
	ax1.plot(s0 * c_table['features'],c_table['test_accuracy'],label=c_names_list[n],color=color_list[n],linestyle='-')

	c_table = d_table[d_table['classifier_id']==c_id]
	ax2.plot(s1* c_table['features'],c_table['test_accuracy'],label=c_names_list[n],color=color_list[n],linestyle='--')

plt.ylabel('test accuracy')
if proportional_features:
	plt.xlabel('proportion of features used')
else:
	plt.xlabel('# features used')
plt.show()
