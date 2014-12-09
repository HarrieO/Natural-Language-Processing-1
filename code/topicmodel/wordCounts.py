import numpy as np
import re, string, random, time
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
from datapoint import *
from scipy.misc import logsumexp
pattern = re.compile('[\W_]+', re.UNICODE)

def giveLabel(score):
    if score > 0.0:
        return 1
    return 0
def get_index(label):
    if label ==2:
        return 1
    return 0
def pickIndexToLogProb(probs):
    ranNum = np.random.rand()*np.sum(probs)
    for i,prob in enumerate(probs):
        ranNum -= prob
        if ranNum <=0:
            return i
    print "Probs are ", probs
    print "sum is ", np.sum(probs)
    return 0

class WordCounter(object):
    def __init__(self, data, giveLabel, alpha=(0.5,0.5),beta=0.5):
        V = np.zeros((3,1)) 

        wordmap           = {}
        totalScorePerWord = {}
        occurencesPerWord = {}
        invwordmap        = [] 
        vocabularySize    = 0
        numberOfSentences = len(data)
        labelsPerSentence = np.zeros((numberOfSentences,1))
        labelCount        = np.zeros((2,1))
        tagCount          = np.zeros(2)

        sentenceWords = []
        data = data[:100]
        for i, post in enumerate(data):
            # raw Strings
            rawWords   = post.content.split()
            label      = giveLabel(post.score)
            labelsPerSentence[i] = label
            labelCount[label]   += 1
            # indices of words after removal of non alphanumeric characters
            cleanWords  = []
            for m,word in enumerate(post.content.split()):
                word = pattern.sub('', word).lower()
                if len(word) > 0:
                    if word not in wordmap:
                        invwordmap.append(word)
                        wordmap[word]   = vocabularySize
                        vocabularySize += 1
                    cleanWords.append(wordmap[word])
                    totalScorePerWord[wordmap[word]] = totalScorePerWord.get(wordmap[word],0) +post.score
                    occurencesPerWord[wordmap[word]] = occurencesPerWord.get(wordmap[word],0) + 1
            sentenceWords.append(np.array(cleanWords))

        averageScorePerWord = np.zeros((vocabularySize,1))
        for word in wordmap.values():
          averageScorePerWord[word] = totalScorePerWord[word]/float(occurencesPerWord[word])

        sentenceTags  = []
        for i, words in enumerate(sentenceWords):
            label = labelsPerSentence[i,0]
            taggedWords = []
            for m,word in enumerate(words):
                if abs(averageScorePerWord[word]) > 0.3:
                    z = label
                else:
                    z = 2
                V[z] += 1
                taggedWords.append(z)
                tagCount[get_index(z)]+=1
            sentenceTags.append(np.array(taggedWords)) # z tags in sentence

        tagsPerWord     = np.zeros((3,vocabularySize))
        tagsPerSentence = np.zeros((2,numberOfSentences))
        for (i, (words, tags)) in enumerate(zip(sentenceWords,sentenceTags)):
            for word, tag in zip(words,tags):
                tagsPerWord[tag,word]  += 1 
                tagsPerSentence[get_index(tag),i] += 1 

        self.sentenceWords       = sentenceWords
        self.sentenceTags        = sentenceTags
        self.tagsPerWord         = tagsPerWord # number of z tags per word
        self.tagsPerSentence     = tagsPerSentence # number of z tags per sentences
        self.sentenceTags        = sentenceTags
        self.V                   = V # stores labels for z, 0 = negative, 1 = positive, 2 = neutral
        self.labelCount          = labelCount # C(0) = #negative sentences, C(1) = #positive sentences
        self.numberOfSentences   = numberOfSentences
        self.vocabularySize      = vocabularySize
        self.wordmap             = wordmap # word -> index for invwordmap
        self.invwordmap          = invwordmap # all words
        self.labelsPerSentence   = labelsPerSentence
        self.averageScorePerWord = averageScorePerWord
        self.tagCount            = tagCount
        self.alpha               = alpha
        self.beta                = beta

    # convert indices back to words
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
        self.tagsPerSentence = np.concatenate((self.tagsPerSentence,np.zeros((2,1))),axis=1)
        self.sentenceTags.append(tags)
        self.labelsPerSentence = np.append(self.labelsPerSentence,np.array([label]))
        self.labelCount[label] += 1
        for i, word in enumerate(sent):
            z       = random.choice([2]+[label])
            self.V[z]   += 1
            tags[i] = z
            self.tagsPerWord[z,word]  += 1
            self.tagsPerSentence[get_index(z),self.numberOfSentences] += 1
            self.tagCount[get_index(z)]+=1
        self.numberOfSentences += 1
        self.sentenceWords.append(sent)
        self.sentenceTags.append(tags)

    def changeLabel(self,sentence_i,word_i,label):
        oldtag = self.sentenceTags[sentence_i][word_i]
        if oldtag != label:
            word = self.sentenceWords[sentence_i][word_i]
            self.V[oldtag]                          -= 1
            self.tagsPerWord[oldtag,word]           -= 1
            #self.tagsPerSentence[get_index(oldtag),sentence_i] -= 1 ###currently used variable
            self.tagCount[get_index(oldtag)]-=1
            self.sentenceTags[sentence_i][word_i]    = label
            self.V[label]                           += 1
            self.tagsPerWord[label,word]            += 1
            #self.tagsPerSentence[get_index(label),sentence_i]  += 1 ###currently used variable
            self.tagCount[get_index(label)]+=1

    # Topic model
    # 0 = negative, 1 = positive, 2 = neutral
    # return X_i ~ p(z_i|Z_{-i}=X_{-i}))
    def conditional_distribution(self,dimension):
        (n,m) = dimension
        currentWord = self.sentenceWords[n][m]
        currentTag = self.sentenceTags[n][m]
        probs = np.zeros(3) # probabilty for labels 0,1,2
        total = self.alpha[0]+self.alpha[1]+self.tagCount[0]+self.tagCount[1]-1.0
        for i in [self.labelsPerSentence[n][0],2]:
            delta = get_index(i)
            if currentTag == i:
                deltaVal=1.0
            else:
                deltaVal = 0.0
            value = (self.tagCount[delta]+self.alpha[delta]-1.0)/total
            Vi = sum(self.tagsPerWord[i,:]>0)
            probs[i] = value*(self.beta -deltaVal +self.tagsPerWord[i,currentWord])/(-deltaVal+self.V[i]+Vi*self.beta)
        newLabel = pickIndexToLogProb(probs)
        self.changeLabel(n,m,newLabel)
        return newLabel


print "Opening data"

data = read_data("../../datasets/preprocessed/trainset.csv")
data.extend(read_data("../../datasets/preprocessed/testset.csv"))

print "Read data starting count"

start = time.time()
wordCounter = WordCounter(data,giveLabel)

print "Took", time.time()-start, "seconds"


def gibbs_sample_topic_model(wordCounter, interestSentInd,num_its=2):
    # burn = number of iterations used for burn-in
    samples_out = []
    X = wordCounter.sentenceTags
    for i in range(num_its):
        for n in range(len(X)):
            for m in range(len(X[n])):
                wordCounter.conditional_distribution((n,m))
        samples_out.append(wordCounter.sentenceTags[interestSentInd])
    return samples_out
print "Labeling before Gibss sampling: "
for i in range(10):
    print " ".join(wordCounter.getWords(wordCounter.sentenceWords[i]))
    print wordCounter.sentenceTags[i]

start = time.time()
num_its= 10
print "Applying Gibbs sampling for", num_its, "iterations"
gibbs_sample_topic_model(wordCounter, 5, num_its)
print "Took", time.time()-start, "seconds"
print "Labeling after Gibss sampling: "
for i in range(10):
    print " ".join(wordCounter.getWords(wordCounter.sentenceWords[i]))
    print wordCounter.sentenceTags[i]
