import numpy as np
import re, string, random, time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
from datapoint import *
pattern = re.compile('[\W_]+', re.UNICODE)

def giveLabel(score):
    if score > 0.5:
        return 1
    elif score < -0.5:
        return -1
    return 0

class WordCounter(object):
    def __init__(self, data, giveLabel):
        V = np.zeros((3,1))

        wordmap           = {}
        invwordmap        = []
        vocabularySize    = 0
        numberOfSentences = len(data)
        labelsPerSentence = np.zeros((numberOfSentences,1))
        labelCount = np.zeros((3,1))

        sentenceWords = []
        sentenceTags  = []
        for i, post in enumerate(data):
            # raw Strings
            rawWords   = post.content.split()
            label      = giveLabel(post.score)
            labelsPerSentence[i] = label
            labelCount[label]   += 1
            # indices of words after removal of non alphanumeric characters
            cleanWords  = []
            taggedWords = []
            for word in post.content.split():
                word = pattern.sub('', word).lower()
                if len(word) > 0:
                    if word not in wordmap:
                        invwordmap.append(word)
                        wordmap[word]   = vocabularySize
                        vocabularySize += 1
                    cleanWords.append(wordmap[word])
                    z     = random.choice([0]+[label])
                    V[z] += 1
                    taggedWords.append(z)
            sentenceWords.append(np.array(cleanWords))
            sentenceTags.append(np.array(taggedWords))

        tagsPerWord     = np.zeros((3,vocabularySize))
        tagsPerSentence = np.zeros((3,numberOfSentences))
        for (i, (words, tags)) in enumerate(zip(sentenceWords,sentenceTags)):
            for word, tag in zip(words,tags):
                tagsPerWord[tag,word]  += 1
                tagsPerSentence[tag,i] += 1

        self.sentenceWords     = sentenceWords
        self.sentenceTags      = sentenceTags
        self.tagsPerWord       = tagsPerWord
        self.tagsPerSentence   = tagsPerSentence
        self.sentenceTags      = sentenceTags
        self.V                 = V
        self.labelCount        = labelCount
        self.numberOfSentences = numberOfSentences
        self.vocabularySize    = vocabularySize
        self.wordmap           = wordmap
        self.invwordmap        = invwordmap
        self.labelsPerSentence = labelsPerSentence

    # converst indices back to words
    def getWords(self,wordArray):
        return [self.invwordmap[i] for i in wordArray]

    def addWord(self,word):
        if word not in self.wordmap:
            self.invwordmap.append(word)
            self.wordmap[word]   = self.vocabularySize
            self.vocabularySize += 1
            self.tagsPerWord = np.concatenate((self.tagsPerWord,np.zeros((3,1))),axis=1)

    # converts string to indices array
    def convertSentence(self, sentence):
        cleanWords = []
        for word in sentence.split():
            word = pattern.sub('', word).lower()
            if len(word) > 0:
                self.addWord(word)
                cleanWords.append(self.wordmap[word])
        return np.array(cleanWords)

    # adds sentence to WordCounter
    def addSentence(self, sentence, label):
        sent = self.convertSentence(sentence)
        tags = np.zeros((len(sent),1))
        self.tagsPerSentence = np.concatenate((self.tagsPerSentence,np.zeros((3,1))),axis=1)
        self.sentenceTags.append(tags)
        self.labelsPerSentence = np.append(self.labelsPerSentence,np.array([[label]]) , axis=0)
        self.labelCount[label] += 1
        for i, word in enumerate(sent):
            z       = random.choice([0]+[label])
            self.V[z]   += 1
            print len(sent), i, tags.shape
            tags[i] = z
            self.tagsPerWord[z,word]  += 1
            self.tagsPerSentence[z,self.numberOfSentences] += 1
        self.numberOfSentences += 1
        self.sentenceWords.append(sent)
        self.sentenceTags.append(tags)

    def changeLabel(self,sentence_i,word_i,label):
        oldtag = self.sentenceTags[sentence_i][word_i]
        if oldtag != label:
            word = self.sentenceWords[sentence_i][word_i]
            self.V[oldtag]                          -= 1
            self.tagsPerWord[oldtag,word]           -= 1
            self.tagsPerSentence[oldtag,sentence_i] -= 1
            self.sentenceTags[sentence_i][word_i]    = label
            self.V[label]                           += 1
            self.tagsPerWord[label,word]            += 1
            self.tagsPerSentence[label,sentence_i]  += 1





print "Opening data"

data = read_data("../discofeatures/trainset.csv")
data.extend(read_data("../discofeatures/testset.csv"))

print "Read data starting count"

start = time.time()
wordCounter = WordCounter(data,giveLabel)

print "Took", time.time()-start, "seconds"

wordCounter.addSentence("This is an example sentence, this is flabbergasted my dickens my dickens, diddy wa diddy.",1)