import numpy as np
import re, string, random, time, pickle
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
from datapoint import *
from scipy.misc import logsumexp
pattern = re.compile('[\W_]+', re.UNICODE)
from wordCounts import *

print "After 0 iterations"
f = open('topicModel.txt', 'r+')
wordCounter = pickle.load(f)
f.close()
"""
for i in range(10):
    print " ".join(wordCounter.getWords(wordCounter.sentenceWords[i]))
    print wordCounter.sentenceTags[i]
print wordCounter.V[0]
print wordCounter.V[1]
"""
print wordCounter.wordTagsForSentence("Would thank you please",1, False)
print wordCounter.wordTagsForSentence("Would you be an idiot!",0, False)
print wordCounter.wordTagsForSentence("Would you be an idiot!",1, False)
#print wordCounter.wordTagsForSentence("You are an idiot!",0, 2)
#print wordCounter.wordTagsForSentence("You are an idiot!",1, 0)
#print wordCounter.wordTagsForSentence("Would you be an idiot!",1, 10)
#print wordCounter.wordTagsForSentence("You are an idiot!",1, 5)

print "After 10000 iterations"
f = open('topicModel10000.txt', 'r+')
wordCounter = pickle.load(f)
f.close()

print wordCounter.wordTagsForSentence("Would you be an idiot!",0, False)
print wordCounter.wordTagsForSentence("Would you be an idiot!",1, False)
print wordCounter.wordTagsForSentence("Would thank you please",1, False)


"""
f = open('topicModel10.txt', 'r+')
wordCounter = pickle.load(f)
f.close()
print "After 10 iterations"
for i in range(10):
    print " ".join(wordCounter.getWords(wordCounter.sentenceWords[i]))
    print wordCounter.sentenceTags[i]
print wordCounter.V[0]
print wordCounter.V[1]
"""