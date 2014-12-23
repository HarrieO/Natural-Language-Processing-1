import post, os, csv
import numpy as np

if os.path.isfile('annotated_data.csv'):
	print 'eee'

else:
	# Subsample not yet taken, take subsample and store file

	sents_to_sample = 3

	train_comments   = post.read_column(0,'train.csv')
	test_comments    = post.read_column(0,'test.csv')

	#take random subsample
	indices = np.array(range(len(train_comments)))
	np.random.shuffle(indices)
	sample_ind = indices[range(sents_to_sample)]


new_ratings = np.zeros(sents_to_sample)

for n,ind in enumerate(sample_ind):
	print '-------------------------------------------------------------------------'
	print 'Please rate the following sentence: [impolite: -3, neutral: 0, polite: 3]'
	print '-------------------------------------------------------------------------'
	print train_comments[ind]
	new_ratings[n] = input()
	print 
	print 



