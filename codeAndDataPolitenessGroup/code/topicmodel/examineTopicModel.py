import numpy as np
import re, string, random, time, pickle
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
from datapoint import *
from scipy.misc import logsumexp
pattern = re.compile('[\W_]+', re.UNICODE)
from wordCounts import *
N = 10
print "N is ", N
f = open('topicModel.txt', 'r+')
wordCounter = pickle.load(f)
f.close()
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
print "N=", N, ", before sampling & "
for i in range(3):
	#print labels[i]
	wordSSorted = np.argsort(averageTagPW[i],)
	wordSSorted = wordSSorted[np.sum(tagPW[:,wordSSorted],0)>N]
	bestWords = []
	for j in range(10):
		bestWords.append(invwordM[wordSSorted[-j-1]])
	print ", ".join(bestWords)
	print " & "

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
print "N=", N, ", after sampling & "
for i in range(3):
	#print labels[i]
	wordSSorted = np.argsort(averageTagPW[i],)
	wordSSorted = wordSSorted[np.sum(tagPW[:,wordSSorted],0)>N]
	bestWords = []
	for j in range(10):
		bestWords.append(invwordM[wordSSorted[-j-1]])
	print ", ".join(bestWords)
	print " & "

print "RANDOM"
N = 300
print "N is ", N
f = open('topicModelRandom.txt', 'r+')
wordCounter = pickle.load(f)
f.close()

tagPW    = wordCounter.tagsPerWord
wordM    = wordCounter.wordmap # word -> index for invwordmap
invwordM = wordCounter.invwordmap # all words
tagPW    = wordCounter.tagsPerWord
wordM    = wordCounter.wordmap # word -> index for invwordmap
invwordM = wordCounter.invwordmap # all words
sumArray = np.array([np.sum(tagPW,0),np.sum(tagPW,0),np.sum(tagPW,0)])
averageTagPW = tagPW / sumArray
labels = ["IMPOLITE", "POLITE", "NEUTRAL"]
print N, "& before & "
for i in range(3):
	#print labels[i]
	wordSSorted = np.argsort(averageTagPW[i],)
	wordSSorted = wordSSorted[np.sum(tagPW[:,wordSSorted],0)>N]
	bestWords = []
	for j in range(10):
		bestWords.append(invwordM[wordSSorted[-j-1]])
	print ", ".join(bestWords)
	print " & "

f = open('topicModelRandom10000000.txt', 'r+')
wordCounter = pickle.load(f)
f.close()

tagPW    = wordCounter.tagsPerWord
wordM    = wordCounter.wordmap # word -> index for invwordmap
invwordM = wordCounter.invwordmap # all words
sumArray = np.array([np.sum(tagPW,0),np.sum(tagPW,0),np.sum(tagPW,0)])
averageTagPW = tagPW / sumArray
labels = ["IMPOLITE", "POLITE", "NEUTRAL"]
print N, "& after & "
for i in range(3):
	#print labels[i]
	wordSSorted = np.argsort(averageTagPW[i],)
	wordSSorted = wordSSorted[np.sum(tagPW[:,wordSSorted],0)>N]
	bestWords = []
	for j in range(10):
		bestWords.append(invwordM[wordSSorted[-j-1]])
	print ", ".join(bestWords)
	print " & "
