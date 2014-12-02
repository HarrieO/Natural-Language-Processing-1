import numpy as np
import post
import collections as Col

def get_counts():
	wordCounts = Col.Counter()
	numberWord = Col.Counter()

	contents    = post.read_column(0,'../../datasets/preprocessed/train.csv')
	scores      = post.read_column(1,'../../datasets/preprocessed/train.csv')
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

# Returns to which class the score belongs
def getClass(score,classCutOff,classes):
	i = 0
	for cutOff in classCutOff:
		if cutOff < score:
			i = i + 1
	return classes[i]


counts = get_counts()
contents  = post.read_column(0,'../../datasets/preprocessed/test.csv')
scores = post.read_column(1,'../../datasets/preprocessed/test.csv')
# for i in range(10):
# 	print contents[i]
# 	print scores[i], " versus ", compute_score(contents[i],counts)
classes			= ['negative','neutral','positive']
classCutOff		= [-0.5,0.5]

misclassifications =0
completeWrong = 0
for i in range(len(contents)):
	classified = getClass(compute_score(contents[i],counts),classCutOff, classes)
	original = getClass(float(scores[i]), classCutOff, classes)
	if classified != original:
		misclassifications +=1
		if not (original == 'neutral' or classified =='neutral'):
			#print contents[i]
			#print classified
			completeWrong +=1
print "Percentage wrong: ", misclassifications/float(len(contents))*100, "%, percentage complete wrong: ", completeWrong/float(len(contents))*100, "%"