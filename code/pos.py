class TagExtractor:

	unigrams = dict()
	bigrams = dict()
	trigrams = dict()
	trigramsEst = dict()
	lexical = dict()

	frequencies = dict()
	tagFrequencies = dict()
	tagFrequencies2 = dict()
	tagFrequencies3 = dict()


	tag1 = "" # Used for tag history
	tag2 = "" # Used for tag history

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
		i = 0
		f = open(self.inputfile, 'r')
		# Read the file line by line
		for line in f:
				# Check for non empty line
				if len(line) > 0:
					if line == self.NEW_LINE:
						# If line is done reset parameters
						self.resetHistory()
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
							i = i + 1
					if i>10000000000000000000000:
						break
		print self.tagFrequencies
		print "Total tokens"
		print self.N
	def calculateProbabilities(self):
		self.tags = self.tagFrequencies.keys()
		for tag in self.tags:
			# Estimate Unigram probability
			self.unigrams[tag] = float(self.tagFrequencies[tag])/float(self.N)
			for tag2 in self.tags:
				# Estimate Bigram probability
				if (tag2+","+tag) in self.tagFrequencies2:
					self.bigrams[tag+"|"+tag2] = float(self.tagFrequencies2[tag2+","+tag])/float(self.tagFrequencies[tag])
				else:
					self.bigrams[tag+"|"+tag2] = 0
				for tag1 in self.tags:
					# Estimate Trigram probability
					if (tag1+","+tag2) in self.tagFrequencies2 and (tag1+","+tag2+","+tag) in self.tagFrequencies3:
						self.trigrams[tag+"|"+tag1+","+tag2] = float(self.tagFrequencies3[tag1+","+tag2+","+tag])/float(self.tagFrequencies2[tag1+","+tag2])
					else:
						self.trigrams[tag+"|"+tag1+","+tag2] = 0
			for word in self.words:
				# Estimate Lexical probability
				if (word+","+tag) not in self.frequencies:
					self.frequencies[word+","+tag] = 0
				self.lexical[word+"|"+tag] = float(self.frequencies[word+","+tag])/float(self.tagFrequencies[tag])
		print "Unigrams"
		print self.unigrams
	def estimateTrigrams(self):
		for key, freq in tagFrequencies3:
			lambda1 = 0
			lambda2 = 0
			lambda3 = 0
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
	def incrFreq(self, key, target):
		if key in target:
			target[key] = target[key]+1
		else:
			target[key] = 1
	def resetHistory(self):
		"""
		Resets tag history
		"""
		self.tag1 = "" 
		self.tag2 = ""


extractor = TagExtractor('../datasets/data for tagging/WSJ02-21.pos')
extractor.calculateProbabilities()