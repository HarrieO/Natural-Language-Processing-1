import numpy as np
import post
from computeEntropy import *
import collections as Col

def get_counts(classCutOff,classes):
	wordCounts = Col.Counter()
	numberWord = Col.Counter()
	wordClass = dict()
	classCount = dict()
	for entry in classes:
		classCount[entry]=0.001 # smoothing

	contents    = post.read_column(0,'train.csv')
	scores      = post.read_column(1,'train.csv')
	scoreArr = np.zeros(len(scores))
	for i in range(len(scores)):
		scoreArr[i] = float(scores[i])
	scoreArr =  scoreArr - np.mean(scoreArr)
	for i in range(len(contents)):
		words = contents[i].split()
		for word in words:
			if not (word in wordClass.keys()):
				wordClass[word]=Col.Counter()
				for entry in classes:
					wordClass[word][entry]=0.00000000001 # smmoothing
			wordClass[word][getClass(scoreArr[i],classCutOff,classes)]+=1
			classCount[getClass(scoreArr[i],classCutOff,classes)]+=1
			wordCounts[word] += scoreArr[i]
			numberWord[word]+=1
	# normalize counts
	for word in numberWord.keys():
		wordCounts[word] = wordCounts[word]/float(numberWord[word])
	wordGain = word_entropy(classCount, wordClass)
	ordered = sorted(wordGain, key=wordGain.get)
	return wordCounts, ordered

def compute_score(sentence, counts):
	words  = sentence.split()
	score = 0.0
	for word in words:
		score += counts[word]
		#print word, ", ", counts[word]
	return score

# Returns to which class the score belongs
def getClass(score,classCutOff,classes):
	i = 0
	for cutOff in classCutOff:
		if cutOff < score:
			i = i + 1
	return classes[i]


classes			= ['negative','neutral','positive']
classCutOff		= [-0.5,0.5]
wordCounts,ordered = get_counts(classCutOff,classes)
contents  = post.read_column(0,'test.csv')
scores = post.read_column(1,'test.csv')
percentages = [0.1,0.25,0.5,0.75,1.0]
for percentage in percentages:
	totalWord = len(wordCounts.keys())
	# throw away x% of the words
	newWordCounts = Col.Counter()
	keepWords = ordered[-int(totalWord*percentage):]
	for word in keepWords:
		newWordCounts[word]=wordCounts[word]
	# for i in range(10):
	# 	print contents[i]
	# 	print scores[i], " versus ", compute_score(contents[i],counts)


	misclassifications =0
	completeWrong = 0
	for i in range(len(contents)):
		classified = getClass(compute_score(contents[i],newWordCounts),classCutOff, classes)
		original = getClass(float(scores[i]), classCutOff, classes)
		if classified != original:
			misclassifications +=1
			if not (original == 'neutral' or classified =='neutral'):
				#print contents[i]
				#print classified
				completeWrong +=1
	print "Fraction of words kept is ", percentage
	print "Percentage right: ", 100.0-misclassifications/float(len(contents))*100, "%, percentage partially correct: ", 100.0-completeWrong/float(len(contents))*100, "%"