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
def split_sentences(txt): return (re.findall(r'(?ms)\s*(.*?(?:\.|\?|!))', txt))  # split sentences

# Read the comments in.
contents    = post.read_column(0,'../train.csv')

#preallocate counter for the number of sentences per datapoint
n_sents = np.ones(len(contents),dtype=int)
sentences = list();

#for each datapoint split the datapoint, add to the file and store the number of sentences
for n,datapoint in enumerate(contents):

	#add the list of sents to the list of sentences and store the number of sents
	sentences.extend(split_sentences(datapoint));
	n_sents[n] = len(split_sentences(datapoint))

#Display statistics
print 'Datapoints: {0} \n Sentences: {1}'.format(len(n_sents),sum(n_sents))

#Store the sentences
file = open("sentences.txt", "w")
[file.write('<s> ' + sent + ' </s>\n') for sent in sentences]
file.close()

#Store the sentence counts
file = open("sent_counts.txt", "w")
[file.write('{0} \n'.format(n)) for n in n_sents]
file.close()

#copy the sentences file into the parser folder:
BLLIP_PATH = '/users/tiesvanrozendaal/NLP/Project/libraries/bllip-parser-BLLIP_ON_MAVERICKS'
MAIN_PATH = os.getcwd()
shutil.copy2('sent_counts.txt', BLLIP_PATH + '/sample-text')
#run the parser
os.chdir(BLLIP_PATH)
call([ './parse.sh', 'sample-text/sample-data.txt'])
os.chdir(MAIN_PATH)


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