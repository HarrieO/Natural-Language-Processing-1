import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re
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

	contents    = post.read_column(0,'../../datasets/preprocessed/train.csv')
	scores      = post.read_column(1,'../../datasets/preprocessed/train.csv')
	scoreArr = np.zeros(len(scores))
	for i in range(len(scores)):
		scoreArr[i] = float(scores[i])
	scoreArr =  scoreArr - np.mean(scoreArr)
	for i in range(len(contents)):
		#words = contents[i].split()
		words = re.findall(r"[\w']+|[.,!?;]",contents[i])
		for word in words:
			#word = word[0]
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
	unused= 0
	for word in words:
		if counts[word]==0:
			unused +=1
		score += counts[word]
		#print word, ", ", counts[word]
	if unused < len(words):
		return score*float((len(words))/(len(words)-unused))

# Returns to which class the score belongs
def getClass(score,classCutOff,classes):
	i = 0
	for cutOff in classCutOff:
		if cutOff < score:
			i = i + 1
	return classes[i]


classes			= [0,1,2]#['negative','neutral','positive']
classCutOff		= [-0.5,0.5]
wordCounts,ordered = get_counts(classCutOff,classes)
contents  = post.read_column(0,'../../datasets/preprocessed/test.csv')
scores = post.read_column(1,'../../datasets/preprocessed/test.csv')
percentages = [0.001,0.05,0.1,0.25,0.5,0.75,1.0]
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
	skip=0

	confusionMatrix = np.zeros([len(classes),len(classes)])

	misclassifications =0
	completeWrong = 0
	for i in range(len(contents)):
		scoreFound = compute_score(contents[i],newWordCounts)
		if scoreFound is None:
			skip +=1
			continue
		classified = getClass(scoreFound,classCutOff, classes)
		original = getClass(float(scores[i]), classCutOff, classes)
		confusionMatrix[classified,original]+=1
		if classified != original:
			misclassifications +=1
			#if not (original == 'neutral' or classified =='neutral'):
				#print contents[i]
				#print classified
				#completeWrong +=1
	print "Fraction of words kept is ", percentage, ", this amounts to ", int(percentage*len(newWordCounts.keys())), " words."
	print "Skipped over ", skip, " sentences"
	print "confusionMatrix is ", confusionMatrix
	print "Percentage right: ", 100.0-misclassifications/float(len(contents)-skip)*100, "%"#, percentage partially correct: ", 100.0-completeWrong/float(len(contents)-skip)*100, "%"