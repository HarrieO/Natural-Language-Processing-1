#!/usr/bin/env python2
# coding=utf-8
import os, glob, sys, re, argparse,shutil
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation
from subprocess import call

# In order to import the main project code
sys.path.append('..')
# import the data reader
import post 

#The tokenizer which splits sentences
def split_sentences(txt): return (re.findall(r'(?ms)\s*(.*?(?:\.|\?|!|$))', txt))  # split sentences

know_issues = [(".NET", "NET"), (".org","org"), (".com","com")]
def replace_know_issues(txt):
	for wi, wo in know_issues:
		txt = (" " + wo + " ").join(txt.split(wi))
	return txt

def replace_urls(txt):
	urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', txt)
	for url in urls:
		txt = " <url> ".join(txt.split(url))
	return txt

# Read the comments in.
contents    = post.read_column(0,'../train.csv')

#preallocate counter for the number of sentences per datapoint
n_sents = np.ones(len(contents),dtype=int)
sentences = list();

#for each datapoint split the datapoint, add to the file and store the number of sentences
for n,datapoint in enumerate(contents):

	# clean data of URLs and sorts
	datapoint = replace_urls(datapoint)
	datapoint = replace_know_issues(datapoint)
	#add the list of sents to the list of sentences and store the number of sents
	splitsents = split_sentences(datapoint)
	corsents = []
	for s in splitsents:
		if len(s) == 1 and (s == "." or s == "," or s == "?" or "!"):
			if len(corsents) > 0:
				corsents[0] += s
			else:
				corsents = [s]
		elif len(s) > 0:
			corsents.append(s)
	sentences.extend(corsents);
	n_sents[n] = len(corsents)

#Display statistics
print 'Datapoints: {0} \n Sentences: {1}'.format(len(n_sents),sum(n_sents))

# store the indices
file = open("test-indices.txt", "w")
[file.write(str(i) +' \n') for i,n in enumerate(n_sents) for _ in range(n)  ]
file.close()

#Store the sentences
file = open("test-sentences.txt", "w")
[file.write('<s> ' + " ".join(sent.split()) + ' </s>\n') for sent in sentences]
file.close()

#Store the sentence counts
file = open("test-sent_counts.txt", "w")
[file.write('{0} \n'.format(n)) for n in n_sents]
file.close()

# #copy the sentences file into the parser folder:
# BLLIP_PATH = '/users/tiesvanrozendaal/NLP/Project/libraries/bllip-parser-BLLIP_ON_MAVERICKS'
# MAIN_PATH = os.getcwd()
# shutil.copy2('sentences.txt', BLLIP_PATH + '/sample-text')
# #run the parser
# os.chdir(BLLIP_PATH)
# call([ './parse.sh', 'sample-text/sentences.txt'])
# os.chdir(MAIN_PATH)


#Per datapunt:
#> Vind elke zin ? . NOT .” ?”
#> Zet elke zin binnen <s> </s> TAGS
#> Elke zin een regel
#> Maak een index

#> Disco dop op file met zinnen (featurespace)

#> Zie tutorial hoe deze features per datapunt
 
#Enters weghalen:
#“ “.join(x.split())

print 'Goodbye world'