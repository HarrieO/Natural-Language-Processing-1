import re
import post
import numpy as np

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
		wordTags = re.findall("(\(([a-zA-Z0-9])* ([a-zA-Z0-9])*\))",tree)
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

def getClass(score,classCutOff,classes):
	i = 0
	for cutOff in classCutOff:
		if cutOff < score:
			i = i + 1
	return classes[i]

def registerScore(word,score,totalScores):
	if word not in totalScores:
		totalScores[word] = 0
	totalScores[word] = totalScores[word] + score

def registerCount(word,wordClass,wordTagCount,classCount,classes):
	if word not in wordTagCount:
		wordTagCount[word] = emptyClassCount(classes)
	wordTagCount[word][wordClass] = wordTagCount[word][wordClass] + 1
	# Increment class count
	classCount[wordClass] = classCount[wordClass]+1

def emptyClassCount(classes):
	classCount = dict()
	for className in classes:
		if className == 'neutral':
			continue
		classCount[className] = 0.0001 #NaN fix
	return classCount

def average_scores(totalScores, wordClassCount):
	scores = dict()
	for word, score in totalScores.iteritems():
		scores[word] = score/sum([count[1] for count in wordClassCount[word].items()])
	return scores

# Returns the information gain for the input for the counts of each binary feature
def word_entropy(classCount, wordClassCount):
	# classCount = dictionary containing counts per class
	# wordClassCount = dictionary containing dictionaries for each feature containing counts per class
	classes = classCount.keys()
	numClasses = len(classes)
	countPerClass = np.zeros(numClasses)
	for i in range(numClasses):
		countPerClass[i]= classCount[classes[i]]
	totalSentences = np.sum(countPerClass)
	initialEntropy = entropy(countPerClass)

	wordGain = dict() # information gain when splitting on the word
	words = wordClassCount.keys()
	for word in words:
		countWordPerClass = np.zeros(numClasses)
		for i in range(numClasses):
			countWordPerClass[i]= wordClassCount[word][classes[i]]
		probWord = np.sum(countWordPerClass)/totalSentences
		wordEntropy= entropy(countWordPerClass)*probWord +(1.0-probWord) *entropy(countPerClass-countWordPerClass)
		wordGain[word] = initialEntropy-wordEntropy 
	return wordGain

def entropy(countPerClass):
	probs = countPerClass/np.sum(countPerClass)
	return -np.sum(probs*np.log(probs))

# Data structures
classes			= ['negative','neutral','positive']
classCutOff		= [-0.5,0.5]
classCount 		= emptyClassCount(classes)
wordTagCount 	= dict()
totalScores		= dict()

# Running starts here
extract('disco/discotrain.csv',classes,classCutOff,wordTagCount,classCount,totalScores);
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
