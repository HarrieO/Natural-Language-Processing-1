import csv
import sys
import numpy as np
import random

random.seed(0)
maxInt = int(sys.maxsize/(10**10))
csv.field_size_limit(maxInt)

def read_column(colNr, fileName):
	with open(fileName) as csvfile:
		readerObject = csv.reader(csvfile, delimiter=',')
		row = next(readerObject)
		column = []
		for row in readerObject:
			counter = 0
			for col in row:
				if counter == colNr:
					column.append(col)
					break
				counter = counter+1
		return column

def write_csv(data,fileName):
	with open(fileName, 'wb') as csvfile:
	    writerObject = csv.writer(csvfile, delimiter=',')
	    for row in data:
	    	writerObject.writerow(row)

def make_list(col1,col2,col3, indices):
	combinedList = []
	for index in indices:
		combinedList.append([col1[index], col2[index], col3[index]])
	return combinedList

# Read all data
#sentencesSE  = np.array(read_column(2,'stack-exchange.annotated.csv')) # sentences
scoresSElist = read_column(13,'../datasets/Stanford politeness corpus/stack-exchange.annotated.csv') # mean score
scoresSE     = np.zeros(len(scoresSElist))
for i in range(len(scoresSElist)):
	scoresSE[i] = scoresSElist[i]
turkScoresSE = np.zeros([5,len(scoresSElist)])
for i in [3,4,5,6,7]:
	turkListSE = read_column(i,'../datasets/Stanford politeness corpus/stack-exchange.annotated.csv') # mean score
	for k in range(len(turkListSE)):
		turkScoresSE[i-3,k]= turkListSE[k]
#sentencesWIK  = np.array(read_column(2,'wikipedia.annotated.csv')) # sentences
scoresWIKlist = read_column(13,'../datasets/Stanford politeness corpus/wikipedia.annotated.csv') # mean score
scoresWIK     = np.zeros(len(scoresWIKlist))
for i in range(len(scoresWIKlist)):
	scoresWIK[i] = scoresWIKlist[i]
turkScoresWIK = np.zeros([5,len(scoresWIKlist)])
for i in [3,4,5,6,7]:
	turkListWIK = read_column(i,'../datasets/Stanford politeness corpus/wikipedia.annotated.csv') # mean score
	for k in range(len(turkListWIK)):
		turkScoresWIK[i-3,k]= turkListWIK[k]

def classFromScore(score):
	if score>0.4548283398341303: 
		return 1
	elif score <-0.38765975611068004:
		return -1
	else:
		return 0
def classFromTurkScore(score, left = 8, right =17):
	if score < left:
		return -1
	elif score > right:
		return 1
	else:
		return 0 
def testTurksRandom(left=8, right =17):
	mistakes = 0
	for i in range(len(scoresSElist)):
		turkIndex = random.randint(0,4)
		if( classFromTurkScore(turkScoresSE[turkIndex, i], left, right)!= classFromScore(scoresSE[i])):
			mistakes +=1
	for i in range(len(scoresWIKlist)):
		turkIndex = random.randint(0,4)
		if( classFromTurkScore(turkScoresWIK[turkIndex, i], left, right)!= classFromScore(scoresWIK[i])):
			mistakes +=1
	percentage = (1.0-mistakes / float(len(scoresSElist)+len(scoresWIKlist))) * 100.0
	print "For split (",left, ",", right, "), ", percentage, "% was classified correctly"
	return percentage
def testTurksAverage(left=8, right =17):
	mistakes = 0
	for i in range(len(scoresSElist)):
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresSE[turkIndex, 0], left, right)!= classFromScore(scoresSE[i])):
				mistakes +=1
	for i in range(len(scoresWIKlist)):
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresWIK[turkIndex, i], left, right)!= classFromScore(scoresWIK[i])):
				mistakes +=1
	percentage = (1.0- 0.2*mistakes / float(len(scoresSElist)+len(scoresWIKlist))) * 100.0
	print "For split (",left, ",", right, "), ", percentage, "% was classified correctly"
	return percentage

def testTurksAll(left=11, right =18):
	allCorrect = 0
	for i in range(len(scoresSElist)):
		mistake = False
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresSE[turkIndex, i], left, right)!= classFromScore(scoresSE[i])):
				mistake = True
		if not mistake:
			allCorrect+=1
	for i in range(len(scoresWIKlist)):
		mistake = False
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresWIK[turkIndex, i], left, right)!= classFromScore(scoresWIK[i])):
				mistake = True
		if not mistake:
			allCorrect+=1
	percentage = (1.0*allCorrect / float(len(scoresSElist)+len(scoresWIKlist))) * 100.0
	print "For split (",left, ",", right, "), ", percentage, "% was classified correctly by all Turks"
	return percentage

def testTurksAgree(left=11, right =18):
	allCorrect = 0
	for i in range(len(scoresSElist)):
		mistake = False
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresSE[turkIndex, i], left, right)!= classFromTurkScore(turkScoresSE[0, i], left, right)):
				mistake = True
		if not mistake:
			allCorrect+=1
	for i in range(len(scoresWIKlist)):
		mistake = False
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresWIK[turkIndex, i], left, right)!= classFromTurkScore(turkScoresWIK[0, i], left, right)):
				mistake = True
		if not mistake:
			allCorrect+=1
	percentage = (1.0*allCorrect / float(len(scoresSElist)+len(scoresWIKlist))) * 100.0
	print "For split (",left, ",", right, "), ", percentage, "% was classified the same by all Turks"
	return percentage

def testTurksCorrectGivenAgree(left=11, right =18):
	testedInstances = 0
	correct = 0
	for i in range(len(scoresSElist)):
		mistake = False
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresSE[turkIndex, i], left, right)!= classFromTurkScore(turkScoresSE[0, i], left, right)):
				mistake = True
		if not mistake: 
			testedInstances +=1
			if (classFromTurkScore(turkScoresSE[0, i], left, right) == classFromScore(scoresSE[i])):
				correct+=1
	for i in range(len(scoresWIKlist)):
		mistake = False
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresWIK[turkIndex, i], left, right)!= classFromTurkScore(turkScoresWIK[0, i], left, right)):
				mistake = True
		if not mistake: 
			testedInstances +=1
			if (classFromTurkScore(turkScoresWIK[0, i], left, right) == classFromScore(scoresWIK[i])):
				correct+=1	
	percentage = (1.0*correct / float(testedInstances)) * 100.0
	occured = (1.0*testedInstances / float(len(scoresSElist)+len(scoresWIKlist))) * 100.0
	print "For split (",left, ",", right, "), ", percentage, "% was classified correctly by all Turks, given they gave the same label, which occured ", occured, "% of the time."
	return percentage, occured

def giveDataPercentages():
	neutral = 0
	positive = 0
	negative = 0
	for i in range(len(scoresSElist)):
		if scoresSE[i]>0.5: 
			positive +=1
		elif scoresSE[i]<-0.5:
			negative+=1
		else:
			neutral +=1
	print "Neutral: ", 100.0*neutral/len(scoresSElist)
	print "Polite: ", 100.0*positive/len(scoresSElist)
	print "Impolite: ", 100.0*negative/len(scoresSElist)

def testTurksAverageOnEqualSet(left=8, right =17):
	examples = np.zeros(3)
	mistakes = np.zeros(3)
	for i in range(len(scoresSElist)):
		label = classFromScore(scoresSE[i])
		examples[label] +=1
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresSE[turkIndex, 0], left, right)!= label):
				mistakes[label] +=1
	for i in range(len(scoresWIKlist)):
		examples[label] +=1
		label = classFromScore(scoresWIK[i])
		for turkIndex in [0,1,2,3,4]:
			if( classFromTurkScore(turkScoresWIK[turkIndex, i], left, right)!= label):
				mistakes[label] +=1
	percentage = (1.0- 0.2* mistakes / examples) * 100.0
	print "For split (",left, ",", right, "), ", percentage, "% was classified correctly, with mean ", np.mean(percentage)
	return np.mean(percentage)


#giveDataPercentages()
#testTurksAverageOnEqualSet()
#testTurksAll()
#testTurksAgree()


print "Evaluating average turk accuracy on equal sets"
bestI = 2
bestJ = 13
bestPercentage = 0
for i in [6,7,8,9,10,11,12]:
	for j in [13,14,15,16,17,18,19,20,22]:
		percentage = testTurksAverageOnEqualSet(i,j)
		if percentage > bestPercentage:
			bestPercentage = percentage
			bestI = i
			bestJ = j
for i in [13,14,15]:
	for j in [16,17,18,19,20,22]:
		percentage = testTurksAverageOnEqualSet(i,j)
		if percentage > bestPercentage:
			bestPercentage = percentage
			bestI = i
			bestJ = j
print "Best (i,j) is ", (bestI,bestJ), ", best percentage is ", bestPercentage


print "Evaluating average turk accuracy on 25-50-25 split"
bestI = 2
bestJ = 13
bestPercentage = 0
for i in [6,7,8,9,10,11,12]:
	for j in [13,14,15,16,17,18,19,20,22]:
		percentage = testTurksAverage(i,j)
		if percentage > bestPercentage:
			bestPercentage = percentage
			bestI = i
			bestJ = j

print "Best (i,j) is ", (bestI,bestJ), ", best percentage is ", bestPercentage

bestI = 2
bestJ = 13
bestPercentage = 0
bestF1 = 0
bestIF1 = 2
bestJF1 = 13
for i in [6,7,8,9,10,11,12]:
	for j in [13,14,15,16,17,18,19,20,22]:
		percentage, occured = testTurksCorrectGivenAgree(i,j)
		f1 = 2.0*percentage*occured/(percentage+occured)
		if percentage > bestPercentage:
			bestPercentage = percentage
			bestI = i
			bestJ = j
		if f1 > bestF1:
			bestF1 = f1
			bestIF1 = i
			bestJF1 = j
print "Best (i,j) is ", (bestI,bestJ), ", best percentage is ", bestPercentage
print "Best (i,j) is ", (bestIF1,bestJF1), ", best F1 is ", bestF1

