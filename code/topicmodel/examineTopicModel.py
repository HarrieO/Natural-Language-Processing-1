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
# print wordCounter.wordTagsForSentence("Would thank you please",1, False)
# print wordCounter.wordTagsForSentence("Would you be an idiot!",0, False)
# print wordCounter.wordTagsForSentence("Would you be an idiot!",1, False)
#print wordCounter.wordTagsForSentence("You are an idiot!",0, 2)
#print wordCounter.wordTagsForSentence("You are an idiot!",1, 0)
#print wordCounter.wordTagsForSentence("Would you be an idiot!",1, 10)
#print wordCounter.wordTagsForSentence("You are an idiot!",1, 5)

tagPW    = wordCounter.tagsPerWord
wordM    = wordCounter.wordmap # word -> index for invwordmap
invwordM = wordCounter.invwordmap # all words
tagPW    = wordCounter.tagsPerWord
wordM    = wordCounter.wordmap # word -> index for invwordmap
invwordM = wordCounter.invwordmap # all words
sumArray = np.array([np.sum(tagPW,0),np.sum(tagPW,0),np.sum(tagPW,0)])
averageTagPW = tagPW / sumArray
labels = ["IMPOLITE", "POLITE", "NEUTRAL"]
for i in range(3):
	print labels[i]
	wordSSorted = np.argsort(averageTagPW[i],)
	wordSSorted = wordSSorted[np.sum(tagPW[:,wordSSorted],0)>20]
	bestWords = []
	for j in range(10):
		bestWords.append(invwordM[wordSSorted[-j-1]])
	print bestWords

print "After 10000000 iterations"
f = open('topicModel10000000.txt', 'r+')
wordCounter = pickle.load(f)
f.close()

# print wordCounter.wordTagsForSentence("Would you be an idiot!",0, False)
# print wordCounter.wordTagsForSentence("Would you be an idiot!",1, False)
# print wordCounter.wordTagsForSentence("Would thank you please",1, False)

tagPW    = wordCounter.tagsPerWord
wordM    = wordCounter.wordmap # word -> index for invwordmap
invwordM = wordCounter.invwordmap # all words
sumArray = np.array([np.sum(tagPW,0),np.sum(tagPW,0),np.sum(tagPW,0)])
averageTagPW = tagPW / sumArray
labels = ["IMPOLITE", "POLITE", "NEUTRAL"]
for i in range(3):
	print labels[i]
	wordSSorted = np.argsort(averageTagPW[i],)
	wordSSorted = wordSSorted[np.sum(tagPW[:,wordSSorted],0)>20]
	bestWords = []
	for j in range(10):
		bestWords.append(invwordM[wordSSorted[-j-1]])
	print bestWords

for word in ['please', 'good', 'why', 'thank', 'homework']:
	print word, ": ", averageTagPW[:,wordM[word]], "; ",tagPW[:,wordM[word]]

