#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse, shutil, nltk.data
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
from subprocess import call

# In order to import the main project code
sys.path.append('..')
# import the data reader
import post 

#The tokenizer which splits sentences
def split_sentences(txt):
	
	#conver unicode (special quotation marks etc.) tp ASCII
	#punctuation = { 0x2018:0x27, 0x2019:0x27, 0x201C:0x22, 0x201D:0x22 }
	#txt = txt.translate(punctuation).encode('ascii', 'ignore')

	#this regex doesn deal with ... .NET and com.apple.bla
	return re.findall(r'(?ms)\s*(.*?(?:\.|\?|!|$))', txt)
	#this regex returns tuples so work around this :P
	#return [sent[0] for sent in re.findall(r'(?ms)\s*(.*?(?:\.|\?|!|$))(\s|$)', txt)]

	return nltk.data.load('tokenizers/punkt/english.pickle').tokenize(txt)


def replace_urls(txt):
	urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
	for url in urls:
		txt = " <url> ".join(txt.split(url))
	return txt

def get_trees(comments):
	"""
	Splits each individual comment into sentences and parses the trees.
	Returns a list of trees for the individual sentences as well as a list of comment indices for each sent. 
	Returns an array of trees
	"""

	#preallocate empty
	sentences = list();
	comment_indices = list();

	#for each datapoint split the datapoint, add to the file and store the number of sentences
	for n,datapoint in enumerate(comments):
		# clean data of URLs
		#datapoint = replace_urls(datapoint)
		#add the list of sents to the list of sentences and store the number of sents
		new_sents = split_sentences(datapoint)
		sentences.extend(new_sents);
		comment_indices.extend([n]*len(new_sents))

	return sentences, comment_indices

def main():
	# Read the comments in.
	comments    = post.read_column(0,'../test.csv')

	sentences, comment_indices = get_trees(comments)

	#Display statistics
	print 'Datapoints: {0} \n Sentences: {1}'.format(len(comment_indices),len(set(comment_indices)) )

	#Store
	file = open("test_comment_indices.txt", "w")
	[file.write('{0} \n'.format(n)) for n in comment_indices]
	file.close()

	#Store the sentences
	file = open("test_sentences.txt", "w")
	for sent in sentences:
		if sent:
		 	file.write('<s> ' + " ".join(sent.split()) + ' </s>\n')


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