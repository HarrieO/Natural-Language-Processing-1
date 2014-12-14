import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import re
import numpy as np
import post
import cPickle as pickle
from computeEntropy import *
import collections as Col

def round_sig_unsigned(x, sig=2):
	x= float(np.abs(x))
	if int(np.floor(np.log10(np.abs(x))))<=-2:
		# use scientific notations
		number =  ("{:."+str(sig-1)+"E}").format(x)
		#if x<0:
		#number *=-1
		return number
	return np.round(x, sig-int(np.floor(np.log10(np.abs(x))))-1)

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
	ordered = sorted(wordGain, key=wordGain.get, reverse=True)
	reverseOrdered = sorted(wordGain, key=wordGain.get)
	return wordCounts,wordGain, ordered, reverseOrdered

def compute_score(sentence, counts):
	# words  = sentence.split()
	words = re.findall(r"[\w']+|[\W]",sentence)
	score = 0.0
	unused= 0
	for word in words:
		if not word in counts or counts[word]==0:
			unused +=1
		else:
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

def sign(number):
	if number<0:
		return "-"
	else:
		return ""

def writeLatexTable(listOfDicts, captions, num_items, keyNums, sig):
	print "\\begin{table}"
	print "\centering"
	print "\\begin{tabular}{|"+("c|"*len(listOfDicts))+"}"
	print "\\hline"
	captionText = ""
	for i in range(len(captions)-1):
		captionText+= "\\textbf{"+captions[i]+"}"
		captionText+= " & "
	captionText+= "\\textbf{"+captions[-1]+"}"
	print captionText + " \\\\"
	print "\\hline"

	keysStart = listOfDicts[keyNums[0]]
	keysEnd = listOfDicts[keyNums[1]]
	for i in range(num_items):
		writeText = ""
		writeText += str(keysStart[i]) + " & "
		for j in range(1,keyNums[1]):
			num = (listOfDicts[j])[keysStart[i]]
			writeText += sign(num)+str(round_sig_unsigned(num,sig)) + " & "
		writeText += str(keysEnd[i]) + " & "
		for j in range(keyNums[1]+1, len(listOfDicts)-1):
			num = (listOfDicts[j])[keysEnd[i]]
			writeText += sign(num)+str(round_sig_unsigned(num,sig)) + " & "
		num =(listOfDicts[-1])[keysEnd[i]]
		writeText += sign(num)+str(round_sig_unsigned(num,sig)) + " \\\\"
		print writeText
	print "\\hline"
	print "\\end{tabular}"
	print "\\caption{caption}"
	print "\\end{table}"



if __name__ == "__main__":
	classes			= [0,1,2]#['negative','neutral','positive']
	classCutOff		= [-0.5,0.5]
	print "Get counts"
	wordCounts,wordGain,ordered, reverseOrdered = get_counts(classCutOff,classes)
	print "Done getting counts"
	file = open("../../datasets/preprocessed/informationGainWords.txt", "w")
	for feature in ordered:
	 	file.write(str(feature)+','+str(wordCounts[feature])+','+str(wordGain[feature])+ '\n')
	file.close()
	writeLatexTable([ordered, wordCounts, wordGain, reverseOrdered, wordCounts, wordGain], ["Word", "Mean score", "Information gain","Word", "Mean score", "Information gain"], 25, [0,3], 4)



