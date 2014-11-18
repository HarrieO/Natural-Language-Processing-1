#!/usr/bin/env python2
# coding=utf-8
import glob, sys, re, argparse
import numpy as np
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation

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

#for each datapoint split the datapoint, add to the file and store the number of sentences
for n,datapoint in enumerate(contents):
	n_sents[n] = len(split_sentences(datapoint))

#Display statistics
print 'Datapoints: {0} \n Sentences: {1}'.format(len(n_sents),sum(n_sents))




# DISPLAY:
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                   help='The datapoint to display')

    args = parser.parse_args()

    n =  args.integers[0]

    zin = contents[n]
    print zin
    print split_sentences(zin)
    print 'Sentences detected: {0}'.format(len(split_sentences(zin)))

# END OF DISPLAY



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