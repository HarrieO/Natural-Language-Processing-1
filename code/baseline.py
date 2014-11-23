import numpy as np
import post
import collections as Col

def get_countes():
	wordCounts = Col.Counter()
	numberWord = Col.Counter()

	contents    = post.read_column(0,'train.csv')
	scores      = post.read_column(1,'train.csv')
	scoreArr = np.zeros(len(scores))
	for i in range(len(scores)):
		scoreArr[i] = float(scores[i])
	scoreArr =  scoreArr - np.mean(scoreArr)
	for i in range(len(contents)):
		words = contents[i].split()
		for word in words:
			wordCounts[word] += scoreArr[i]
			numberWord[word]+=1
	# normalize counts
	for word in numberWord.keys():
		wordCounts[word] = wordCounts[word]/float(numberWord[word])
	return wordCounts

def compute_score(sentence, counts):
	words  = sentence.split()
	score = 0.0
	for word in words:
		score += counts[word]
		#print word, ", ", counts[word]
	return score

counts = get_countes
