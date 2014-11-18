import numpy as np

def word_entropy(classCount, wordClassCount):
	# classCount = dictionary containing counts per class
	# wordClassCount = dictionary containing dictionaries per word containing counts per class
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


classDict = dict()
classDict[1]=10
classDict[2]=12
wordDict = dict()
wordDict['a']=dict()
wordDict['a'][1]=5
wordDict['a'][2]=3

print word_entropy(classDict,wordDict)