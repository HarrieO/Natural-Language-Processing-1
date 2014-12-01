#!/usr/bin/env python2
# coding=utf-8
import subprocess, os, sys, re, shutil
import numpy as np

# In order to import the main project code
sys.path.append('..')
# import the data reader
import post 

#The tokenizer which splits sentences
def split_sentences(txt):

	know_issues = [(".NET", "NET"), (".org","org"), (".com","com"),('i.e.','ie'),('e.g.','eg')]

	def replace_know_issues(txt):
		for wi, wo in know_issues:
			txt = (" " + wo + " ").join(txt.split(wi))
		return txt

	def replace_urls(txt):
		urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
		for url in urls:
			txt = " <url> ".join(txt.split(url))
		return txt

	# clean data of URLs and sorts
	txt = replace_urls(txt)
	txt = replace_know_issues(txt)

	# perform an preliminary split
	splitsents = re.findall(r'(?ms)\s*(.*?(?:\.|\?|!|$))', txt)

	# fuse lone punctuation back into the sentence
	sentences = []
	for s in splitsents:
		if len(s) == 1 and (s == "." or s == "," or s == "?" or "!"):
			if len(sentences) > 0:
				sentences[0] += s
			else:
				sentences = [s]
		elif len(s) > 0:
			sentences.append(s)

	return sentences


def batch_split_sentences(comments):
	"""
	Splits sentences for a dataset and keeps track of the indices
	"""

	#if a single sentence is given, convert to a single element list
	if isinstance(comments, basestring): comments = [comments]

	#preallocate empty
	sentences = list();
	comment_indices = list();

	#for each datapoint split the datapoint, add to the file and store the number of sentences
	for n,datapoint in enumerate(comments):

		#add the list of sents to the list of sentences and store the number of sents
		new_sents = split_sentences(datapoint)
		sentences.extend(new_sents);
		comment_indices.extend([n]*len(new_sents))

	return sentences, comment_indices

def parse_trees(sentences,prefix,return_trees=1):
	"""
	Stores a sentence- and a tree file. If return_trees=true, the function will wait for the parsing to finish 
	and return a list of trees.
	"""

	BLLIP_PATH = '/users/tiesvanrozendaal/NLP/Project/libraries/bllip-parser-BLLIP_ON_MAVERICKS'

	#Store sentences
	file = open(prefix + '_sentences.txt', 'w')
	[file.write('<s> ' + " ".join(sent.split()) + ' </s>\n') for sent in sentences if sent]
	file.close()


	tree_file = open(prefix + '_trees.txt', 'w')
	MAIN_PATH = os.getcwd()

	#copy the sentences to the parser folder
	shutil.copy2(prefix + '_sentences.txt', BLLIP_PATH + '/sample-text/auto_copied_sents.txt')

	#run the parser
	os.chdir(BLLIP_PATH)
	p = subprocess.Popen(['./parse.sh','sample-text/auto_copied_sents.txt'], stdout=tree_file)
	#call([ './parse.sh', 'sample-text/sentences.txt'])
	os.chdir(MAIN_PATH)
	file.close()

	if return_trees:
		#wait for the process to finish and return the trees in the file
		p.wait()
		return open(prefix + "_trees.txt").readlines()
	

def get_trees(data,prefix):
	"""
	Returns the trees of the data, please define a prefix for the files that are stored.
	If the data is a string, the string will be parsed, if the data is a list, each element will be parsed and 
	a list keeping track of indices will be returned as well.
	"""

	if isinstance(data, basestring):
		sentences = split_sentences(data)
		return parse_trees(sentences,prefix,1)
	else:
		sentences, indices = batch_split_sentences(data)
		return parse_trees(sentences,prefix,1) , indices




def main():
	# Read the comments in.
	train_comments   = post.read_column(0,'../train.csv')
	test_comments    = post.read_column(0,'../test.csv')

	train_sents, train_indices = batch_split_sentences(train_comments)
	test_sents,  test_indices  = batch_split_sentences(test_comments)

	#Display statistics
	print 'Training: Datapoints: {0}, Sentences: {1}'.format(len(train_indices),len(set(train_indices)) )
	print 'Test    : Datapoints: {0}, Sentences: {1}'.format(len(test_indices), len(set(test_indices)) )

	#store indices
	file = open("train_indices.txt", "w")
	[file.write('{0} \n'.format(n)) for n in train_indices]
	file.close()

	file = open("test_indices.txt", "w")
	[file.write('{0} \n'.format(n)) for n in test_indices]
	file.close()

	#parse and store trees
	#parse_trees(train_sents,'train')
	#parse_trees(test_sents,'test')

	#examples
	#print get_trees(test_comments[0:3],'demo')
	#print get_trees(test_comments[0],'demo')
	print parse_trees(train_sents[0:1],'DEMO',1)

if __name__ == '__main__':
	main()

#Store the sentences
#file = open("new_sentences.txt", "w")
# for n, sent in enumerate(sentences):
# 	print sent
# 	print n
# 	file.write('<s> ' + " ".join(sent.split()) + ' </s>\n')

#[file.write('<s> ' + " ".join(sent.split()) + ' </s>\n') for sent in sentences]
#file.close()

#Store the sentence counts
#file = open("comment_indices.txt", "w")
#[file.write('{0} \n'.format(n)) for n in comment_indices]
#file.close()

#copy the sentences file into the parser folder:
#BLLIP_PATH = '/users/tiesvanrozendaal/NLP/Project/libraries/bllip-parser-BLLIP_ON_MAVERICKS'
#MAIN_PATH = os.getcwd()
#shutil.copy2('sentences.txt', BLLIP_PATH + '/sample-text')
#run the parser
#os.chdir(BLLIP_PATH)
#call([ './parse.sh', 'sample-text/sentences.txt'])
#os.chdir(MAIN_PATH)


#“ “.join(x.split())