import re
import post
import os, sys
import numpy as np
import cPickle as pickle
from computeEntropy import word_entropy, entropy

'''
Settings
'''
N = 1000 # The amount of features selected for the histogram

# Computes the word-tag count, the class count and the summed scores for each of the word-tag pairs.
def extract(inputfile,classes,classCutOff,wordTagCount,classCount,totalScores):	
	# inputfile = The file containing the scores
	# classes = List of classes
	# classCutOff = The score value seperating each of the classes (requires: len(classes)-1 == len(classCutOff))
	# wordTagCount = dictionary used to store the word-tag counts
	# classCount = dictionary used to store the class counts
	# totalScores = dictionary used to store the summed score for each word-tag
	i = 0
	scores = map(float,post.read_column(2,inputfile))
	trees = post.read_column(4,inputfile)
	# Read the file line by line
	i = 0
	for tree in trees:
		punctuation = r"[.,!?;]"
		wordTags = re.findall("(\(([a-zA-Z0-9]|"+punctuation+")* ([a-zA-Z0-9]|"+punctuation+")*\))",tree)
		for wordTag in wordTags:
			# Get rid of the brackets and split into word and tag
			#[word, tag] = wordTag[0][1:-1].split(" ")
			#TODO: Do we need to do anything with the tag for smoothing?
			wordClass = getClass(scores[i],classCutOff,classes)
			if wordClass == 'neutral':
				continue
			registerCount(wordTag[0], wordClass, wordTagCount, classCount, classes)
			registerScore(wordTag[0], scores[i], totalScores)
		i = i + 1

# Returns to which class the score belongs
def getClass(score,classCutOff,classes):
	i = 0
	for cutOff in classCutOff:
		if cutOff < score:
			i = i + 1
	return classes[i]


def scoresToClass(scores,classCutOff,classes):
	return [getClass(score,classCutOff,classes) for score in scores]

# Stores the score in the dictionary for total scores
def registerScore(wordTag,score,totalScores):
	if wordTag not in totalScores:
		totalScores[wordTag] = 0
	totalScores[wordTag] = totalScores[wordTag] + score

def registerCount(wordTag,wordClass,wordTagCount,classCount,classes):
	if wordTag not in wordTagCount:
		wordTagCount[wordTag] = emptyClassCount(classes)
	wordTagCount[wordTag][wordClass] = wordTagCount[wordTag][wordClass] + 1
	# Increment class count
	classCount[wordClass] = classCount[wordClass]+1

def emptyClassCount(classes):
	classCount = dict()
	for className in classes:
		if className == 'neutral':
			continue
		classCount[className] = 0.001 #NaN fix
	return classCount

def average_scores(totalScores, wordClassCount):
	scores = dict()
	for word, score in totalScores.iteritems():
		scores[word] = score/sum([count[1] for count in wordClassCount[word].items()])
	return scores

# Returns the counts of the selected wordTags
def selectFeatures(featureEntropy, N, wordTagCount):
	selectedFeatures = dict()
	ordered = sorted(featureEntropy, key=featureEntropy.get)
	for feature in ordered[-N:]:
		selectedFeatures[feature] = wordTagCount[feature]
	ignoredFeatures = [key for key in wordTagCount.keys() if key not in selectedFeatures.keys()]
	return selectedFeatures, ignoredFeatures

def getWordTagsFromTree(tree):
	punctuation = r"[.,!?;]"
	wordTags = re.findall("(\(([a-zA-Z0-9]|"+punctuation+")* ([a-zA-Z0-9]|"+punctuation+")*\))",tree)
	return wordTags

def getHistograms(trees, features):
	# Read the file line by line
	i = 0
	histograms = [0] * len(trees)
	for tree in trees:
		histograms[i] = [0] * len(features)
		wordTags = getWordTagsFromTree(tree)
		for wordTag in wordTags:
			print wordTag
			try:
				idx = features.index(wordTag[0])
			except ValueError:
				idx = -1
			if idx != -1:
				# Include in the histogram
				histograms[i][idx] = histograms[i][idx]+1
		i = i + 1
	return histograms

def outputHistograms(inputfile, outputfile, classes, classCutOff, features):
	trees = post.read_column(4,inputfile)
	scores = map(float,post.read_column(2,inputfile))
	f = open(outputfile, 'w')
	histograms = getHistograms(trees, features);
	i = 0
	for tree in trees:
		# Write the histogram to the file
		f.write(",".join(str(x) for x in histograms[i])+","+getClass(scores[i],classCutOff,classes)+"\n")
		i = i + 1

def reduceFeatureSpace(inputFile,classes,classCutOff,wordTagCount,classCount,totalScores, N):
	extract(inputFile,classes,classCutOff,wordTagCount,classCount,totalScores);
	wordTag_entropy = word_entropy(classCount, wordTagCount)
	newCounts, ignoredWordTags = selectFeatures(wordTag_entropy, N, wordTagCount)
	return newCounts, ignoredWordTags

if __name__ == "__main__":
	# Data structures
	classes			= ['negative','neutral','positive']
	classCutOff		= [-0.5,0.5]
	classCount 		= emptyClassCount(classes)
	wordTagCount 	= dict()
	totalScores		= dict()


	# Running starts here
	extract(os.path.join(os.path.dirname(__file__), '../datasets/preprocessed/discotrain.csv'),classes,classCutOff,wordTagCount,classCount,totalScores);
	wordTag_entropy = word_entropy(classCount, wordTagCount)
	scores = average_scores(totalScores, wordTagCount)
	print "Ordered scores"
	ordered = sorted(wordTag_entropy, key=wordTag_entropy.get)
	orderList=  ordered[-40:]
	scoreList=  [scores[word] for word in ordered[-40:]]
	print zip(orderList,scoreList)
	orderList=  ordered[:40]
	scoreList=  [scores[word] for word in ordered[:40]]
	print zip(orderList,scoreList)
	# Get the counts for the 100 word tags with the highest entropy
	print "Word tag counts"
	newCounts, ignoredWordTags = selectFeatures(wordTag_entropy, N, wordTagCount)
	# Output for highlighting in the demo
	scoreList = dict()
	i = 0
	for word in ordered[:N]:
		scoreList[word.split()[1][:-1]] = scores[word]
		i = i + 1
	pickle.dump(scoreList,open(os.path.join(os.path.dirname(__file__), '../datasets/preprocessed/wordScores.p'),'w+b'))
	outputHistograms(os.path.join(os.path.dirname(__file__), '../datasets/preprocessed/discotrain.csv'), os.path.join(os.path.dirname(__file__), '../datasets/preprocessed/trainHist.csv'), classes, classCutOff, newCounts.keys())
	#print newCounts