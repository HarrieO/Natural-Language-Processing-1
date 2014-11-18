import numpy as np
import collections as col

def specialDivide(x,y):
	if x<=0 and y<=0:
		return 0
	elif x<0:
		return 0
	else: 
		return x/y 

class TagExtractor:
	unigrams 	 = col.Counter()
	bigrams 	 = col.Counter()
	trigrams 	 = col.Counter()
	trigramsEst  = col.Counter() # smoothed version
	wordGivenTag =  dict()
	suffixProb 	 = dict() # reversed suffices

	lexical 	 = col.Counter() # word counts
	lexicalSmall = col.Counter() # word counts for words with count <= 10

	frequencies 	= col.Counter()
	tagFrequencies 	= col.Counter()
	tagFrequencies2 = col.Counter()
	tagFrequencies3 = col.Counter()

	tagSuffFrequencies = dict()
	suffFrequencies 	= col.Counter()

	theta = 0.1

	tag1 = "$" # Used for tag history
	tag2 = "$" # Used for tag history

	tags = []
	words = set()
	N = 0
	# Constants
	NEW_LINE = '======================================'

	def __init__(self, inputfile):
		# Read the file
		self.inputfile = inputfile
		self.extract() # Extract file


	def extract(self):	
		"""
		Extracts tags and words from an input file
		"""
		f = open(self.inputfile, 'r')
		# Read the file line by line
		for line in f:
				# Check for non empty line
				if len(line) > 0:
					if line == self.NEW_LINE:
						# If line is done reset parameters
						self.recordEndTag()
					else:
						# Split in words on one line
						words = line.split(" ")
						for word in words:
							# Ignore the context tags?
							word = word.strip()
							if word != "[" and word != "]":
								# Split in pos tag and word
								items = word.split("/")
								# Get the pos tag
								if len(items)==2:
									[w, t] = items
									# Record the frequencies
									self.recordTag(w, t)
									self.N = self.N + 1
		print self.tagFrequencies
		print "Total tokens: ", self.N
		self.tags = self.tagFrequencies.keys()
		f.close()
	def calculateProbabilities(self):
		for tag in self.tags:
			# Estimate Unigram probability
			self.unigrams[tag] = float(self.tagFrequencies[tag])/float(self.N)
			for tag2 in self.tags:
				# Estimate Bigram probability
				self.bigrams[tag+"|"+tag2] = float(self.tagFrequencies2[tag2+","+tag])/float(self.tagFrequencies[tag2])
				for tag1 in self.tags:
					# Estimate Trigram probability
					if self.tagFrequencies2[tag1+","+tag2] != 0:
						self.trigrams[tag+"|"+tag1+","+tag2] = float(self.tagFrequencies3[tag1+","+tag2+","+tag])/float(self.tagFrequencies2[tag1+","+tag2])
					else:
						self.trigrams[tag+"|"+tag1+","+tag2] = 0
		for word in self.words:
			self.wordGivenTag[word] = dict()
			for tag in self.tags:
				# Estimate Lexical probability and word frequencies
				self.lexical[word] = self.lexical[word]+self.frequencies[word+","+tag]
				self.wordGivenTag[word][tag] =  float(self.frequencies[word+","+tag])/float(self.tagFrequencies[tag])
		print "Unigrams: ", self.unigrams
	def calculateSuffixProbs(self):
		theta = self.theta
		for tag in self.tagFrequencies.keys():
			self.tagSuffFrequencies[tag] = dict()
			for m  in range(10):
				self.tagSuffFrequencies[tag][m]= col.Counter()
		for word in self.lexicalSmall.keys():
			# add lexical counts
			wordList = list(word)
			reverseWord = list(wordList)
			reverseWord.reverse()
			wordTags = self.wordGivenTag[word].keys() # create lists of possible tags for given word
			for tag in wordTags:
				itList = range(min(10, len(reverseWord)))
				for m in itList:
					suff = "".join(list(reverseWord[0:m+1]))
					self.suffFrequencies[suff]+=self.lexicalSmall[word]
					self.tagSuffFrequencies[tag][m][suff]+= self.frequencies[word+"|"+tag]
		for tag in self.tagSuffFrequencies.keys():
			self.suffixProb[tag] = dict()
			prob = self.tagFrequencies[tag]
			for m in range(10):
				if m==0:
					for suff in self.tagSuffFrequencies[tag][m].keys():
						probHat = float(self.tagSuffFrequencies[tag][m][suff])/float(self.suffFrequencies[suff])
						self.suffixProb[tag][suff]=(probHat+theta*prob)/(1.0+theta)
				else:
					for suff in self.tagSuffFrequencies[tag][m].keys():
						self.probHat = float(self.tagSuffFrequencies[tag][m][suff])/float(self.suffFrequencies[suff])
						self.suffixProb[tag][suff]=(probHat+theta*self.suffixProb[tag][suff[0:-1]])/(1.0+theta)
			# Bayesian inversion
			for suff in self.suffixProb[tag].keys():
				self.suffixProb[tag][suff] =float(self.suffFrequencies[suff])/float(self.tagFrequencies[tag])*self.suffixProb[tag][suff]  

	def estimateTrigrams(self):
		lambda1 = 0
		lambda2 = 0
		lambda3 = 0
		for tag1 in self.tags:
			for tag2 in self.tags:
				for tag3 in self.tags:
					ind = np.argmax([specialDivide(self.tagFrequencies3[tag1+","+tag2+","+tag3]-1,self.tagFrequencies2[tag1+","+tag2]-1),
							specialDivide(self.tagFrequencies2[tag2+","+tag3]-1,self.tagFrequencies[tag2]-1),
							specialDivide(self.tagFrequencies[tag1]-1,self.N-1)])
					if ind == 2:
						lambda1 = lambda1 + self.tagFrequencies3[tag1+","+tag2+","+tag3]
					elif ind == 1:
						lambda2 = lambda2 + self.tagFrequencies3[tag1+","+tag2+","+tag3]
					else:
						lambda3 = lambda3 + self.tagFrequencies3[tag1+","+tag2+","+tag3]
		lambdaSum = lambda1+lambda2+lambda3
		lambda1 = lambda1/lambdaSum
		lambda2 = lambda2/lambdaSum
		lambda3 = lambda3/lambdaSum
		# Linear interpolation:
		for tag1 in self.tags:
			for tag2 in self.tags:
				for tag3 in self.tags:
					self.trigramsEst[tag3+"|"+tag1+","+tag2] = lambda1* self.unigrams[tag3] + lambda2*self.bigrams[tag3+"|"+tag2] +lambda3*self.trigrams[tag3+"|"+tag1+","+tag2]
		self.tagFrequencies3 = col.Counter()
		self.tagFrequencies2 = col.Counter()
	def recordTag(self, word, tag):
		# Store the tag counts
		self.incrFreq(tag, self.tagFrequencies)
		# Store f(t2,t3)
		self.incrFreq(self.tag2+","+tag, self.tagFrequencies2)
		# Store f(t1,t2,t3)
		self.incrFreq(self.tag1+","+self.tag2+","+tag, self.tagFrequencies3)
		# Store the word tag combination
		self.incrFreq(word+","+tag, self.frequencies)
		self.words.add(word)
		# Update the tag history
		self.tag1 = self.tag2
		self.tag2 = tag
	def recordEndTag(self):
		# Store the tag counts
		self.incrFreq("$", self.tagFrequencies)
		# Store f(t2,t3)
		self.incrFreq(self.tag2+","+"$", self.tagFrequencies2)
		# Store f(t1,t2,t3)
		self.incrFreq(self.tag1+","+self.tag2+","+"$", self.tagFrequencies3)
		self.resetHistory

	def incrFreq(self, key, target):
		if key in target:
			target[key] = target[key]+1
		else:
			target[key] = 1
	def resetHistory(self):
		"""
		Resets tag history
		"""
		self.tag1 = "$" 
		self.tag2 = "$"
	def estimateTheta(self):
		probs = self.unigrams.values()
		probs = np.array(probs)
		self.theta = np.std(probs)
		print "Theta: ", self.theta
		for word in self.lexical.keys():
			if self.lexical[word] <= 10:
				self.lexicalSmall[word] = self.lexical[word]
		self.lexical = col.Counter() # counts no longer needed!


extractor = TagExtractor('../datasets/data for tagging/WSJ02-21.pos')
extractor.calculateProbabilities()
extractor.estimateTrigrams()
extractor.estimateTheta()
print "Calculating all suffix probabilities..."
extractor.calculateSuffixProbs()
print "Dictionaries: "
transProb = extractor.trigramsEst
emissionProbKnown = extractor.wordGivenTag
suffixProb = extractor.suffixProb

def transProb(tag,given1,given2):
	return trigramsEst[tag+"|"+given1+","+given2]
def emissionProb(word, tag):
	if emissionProbKnown.has_key(word):
		return emissionProbKnown(word)
	else:
		wordList = list(word)
		reverseWord = list(wordList) # copy
		reverseWord.reverse()
		maxM = np.min(10,len(wordList))
		iterateList = range(maxM)
		iterateList.reverse()
		for m in iterateList: # m = maxM-1,..., 0
			if suffixProb[tag].has_key("".join(reverseWord[0:m+1])):
				return suffixProb[tag]["".join(reverseWord[0:m+1])]
