import numpy as np
import re, string
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
from datapoint import *
pattern = re.compile('[\W_]+', re.UNICODE)

data = read_data("../discofeatures/trainset.csv")
data.extend(read_data("../discofeatures/testset.csv"))

# number of latent variables
numberOfClasses = 3

wordmap = {}
vocabularySize = 0

wordsPerSentence = []
for post in data:
    # raw Strings
    rawWords   = post.content.split()
    # indices of words after removal of non alphanumeric characters
    cleanWords = []
    for word in post.content.split():
        word = pattern.sub('', word)
        if len(word) > 0:
            if word not in wordmap:
                wordmap[word]   = vocabularySize
                vocabularySize += 1
            cleanWords.append(wordmap[word])
    wordsPerSentence.append(np.array(cleanWords))