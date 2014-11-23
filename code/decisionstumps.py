import re
import post
import numpy as np

class DecisionStumps:

	classCount 		= dict()
	wordCount 		= dict()
	totalScores		= dict()
	classes			= ['negative','neutral','positive']
	classCutOff		= [-0.5,0.5]

	def __init__(self, inputfile):
		# Read the file
		self.inputfile = inputfile
		self.classCount = self.emptyClassCount()
		self.extract() # Extract file
		print self.classCount


	def extract(self):	
		i = 0
		scores = map(float,post.read_column(2,self.inputfile))
		trees = post.read_column(4,self.inputfile)
		# Read the file line by line
		i = 0
		for tree in trees:
			wordTags = re.findall("(\(([a-zA-Z0-9])* ([a-zA-Z0-9])*\))",tree)
			for wordTag in wordTags:
				# Get rid of the brackets and split into word and tag
				#[word, tag] = wordTag[0][1:-1].split(" ")
				#TODO: Do we need to do anything with the tag for smoothing?
				wordClass = self.getClass(scores[i])
				if wordClass == 'neutral':
					continue
				self.registerCount(wordTag[0], wordClass)
				self.registerScore(wordTag[0], scores[i])
			i = i + 1

	def getClass(self,score):
		i = 0
		for cutOff in self.classCutOff:
			if cutOff < score:
				i = i + 1
		return self.classes[i]

	def registerScore(self,word,score):
		if word not in self.totalScores:
			self.totalScores[word] = 0
		self.totalScores[word] = self.totalScores[word] + score

	def registerCount(self,word,wordClass):
		if word not in self.wordCount:
			self.wordCount[word] = self.emptyClassCount()
		self.wordCount[word][wordClass] = self.wordCount[word][wordClass] + 1
		# Increment class count
		self.classCount[wordClass] = self.classCount[wordClass]+1
	def emptyClassCount(self):
		classCount = dict()
		for className in self.classes:
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

decisionStumps = DecisionStumps('disco/discotrain.csv')
wordTag_entropy = word_entropy(decisionStumps.classCount, decisionStumps.wordCount)
scores = average_scores(decisionStumps.totalScores, decisionStumps.wordCount)
print "Ordered scores"
ordered = sorted(wordTag_entropy, key=wordTag_entropy.get)
orderList=  ordered[-40:]
scoreList=  [scores[word] for word in ordered[-40:]]
print zip(orderList,scoreList)
orderList=  ordered[:40]
scoreList=  [scores[word] for word in ordered[:40]]
print zip(orderList,scoreList)
