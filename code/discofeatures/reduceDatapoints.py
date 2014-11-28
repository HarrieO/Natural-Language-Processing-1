from computeEntropy import *
from datapoint import *
import collections as col

def entropy(countPerClass):
	probs = countPerClass/float(np.sum(countPerClass))
	probs = probs[probs>0.0]
	return -np.sum(probs*np.log(probs))

# Returns to which class the score belongs
def getClass(score,classCutOff,classes):
	i = 0
	for cutOff in classCutOff:
		if cutOff < score:
			i = i + 1
	return classes[i]

def reduceDatapoints(fileName='featureData.csv', classes=[0,1,2], classCutOff=[-0.5,0.5],fractionKept=0.01, start=0, end=270000):
	print "Reading data"
	posts = read_data(fileName)
	print "Finished reading data, started accumulating counts"
	wordCountClasses = np.zeros([len(classes),end])
	classCount = np.zeros(len(classes))
	for post in posts:
		currClass = getClass(float(post.score), classCutOff, classes)
		classCount[currClass]+=1
		wordCountClasses[currClass][np.array(post.fragments.keys())]+=1
	print "Computing entropy per class"
	initialEntropy = entropy(classCount)
	print "Result: ", initialEntropy
	print "Computing information gain per feature for features ", start, " to ",end
	countWord =np.sum(wordCountClasses,0) 
 	probWord = countWord/ float(np.sum(classCount)) # probability a feature occurs in sentence
	dictWord = dict()
	for i in range((countWord[countWord!=0]).shape[0]):
	 	wordEntropy = entropy(wordCountClasses[:,i])*probWord[i] +(1.0-probWord[i]) *entropy(classCount-wordCountClasses[:,i])
		dictWord[i] = wordEntropy-initialEntropy 
	ordered = sorted(dictWord, key=dictWord.get)
	file = open("informationGain.txt", "w")
	for feature in ordered:
	 	file.write(str(feature)+','+str(dictWord[feature])+ '\n')
	file.close()
reduceDatapoints()