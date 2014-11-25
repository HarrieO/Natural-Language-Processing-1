from computeEntropy import *
from datapoint import *

# Returns to which class the score belongs
def getClass(score,classCutOff,classes):
	i = 0
	for cutOff in classCutOff:
		if cutOff < score:
			i = i + 1
	return classes[i]

def reduceDatapoints(fileName='featureData.csv', classes=['negative','neutral','positive'], classCutOff=[-0.5,0.5],fractionKept=0.01):
	print "Reading data"
	posts = read_posts(fileName)
	print "Finished reading data, started accumulating counts"
	classCount = dict()
	for c in classes:
		classCount[c] = 0
	wordClassCount = dict()
	for post in posts:
		words = post.fragments.keys()
		currClass = getClass(float(post.score), classCutOff, classes)
		classCount[currClass]+=1
		for word in words:
			if not(word in wordClassCount.keys()):
				wordClassCount[word]=dict()
				for entry in classes:
					wordClassCount[word][entry] = 0
			wordClassCount[word][currClass]+=1
	print "Computing entropy per feature"
	gainPerFeature = word_entropy(classCount, wordClassCount, True)

	ordered = sorted(gainPerFeature, key=gainPerFeature.get, reverse =True) # list of features ordered by gain
	keepFeatures = ordered[:int(fractionKept*len(ordered))]
	file = open("sortedIndices.txt", "w")
	for feature in ordered:
		file.write(str(feature)+','+str(gainPerFeature[feature])+ '\n')
	file.close()
	return keepFeatures
keepFeatures = reduceDatapoints()