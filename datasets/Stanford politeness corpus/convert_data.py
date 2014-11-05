import csv
import sys
import numpy as np

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

# Read and visualise all data
comminutySE = np.array(read_column(0,'stack-exchange.annotated.csv')) # community
sentencesSE = np.array(read_column(2,'stack-exchange.annotated.csv')) # sentences
scoresSElist = read_column(13,'stack-exchange.annotated.csv') # mean score
scoresSE = np.zeros(len(scoresSElist))
for i in range(len(scoresSElist)):
	scoresSE[i]=scoresSElist[i]
indicesSE = np.array(np.argsort(scoresSE))
print sentencesSE[indicesSE]
print scoresSE[indicesSE]
print comminutySE[indicesSE]
comminutyWIK = np.array(read_column(0,'wikipedia.annotated.csv')) # sentences
sentencesWIK = np.array(read_column(2,'wikipedia.annotated.csv')) # sentences
scoresWIKlist = read_column(13,'wikipedia.annotated.csv') # mean score
scoresWIK = np.zeros(len(scoresWIKlist))
for i in range(len(scoresWIKlist)):
	scoresWIK[i]=scoresWIKlist[i]
indices = np.array(np.argsort(scoresWIK))
print scoresWIK[indices]
print sentencesWIK[indices]

# Distribute into train and test data (indices)
randomIndices = np.random.permutation(indicesSE)
switchSE = int(np.floor(0.8*indicesSE.shape[0]))
trainSE = randomIndices[:switchSE]
testSE = randomIndices[switchSE:]
trainData1 = make_list(sentencesSE,scoresSE,comminutySE, trainSE)
testData1 = make_list(sentencesSE,scoresSE,comminutySE, testSE)
randomIndices = np.random.permutation(indices)
switchWIK = int(np.floor(0.8*indices.shape[0]))
trainWIK = randomIndices[:switchWIK]
testWIK = randomIndices[switchWIK:]
trainData2 = make_list(sentencesWIK,scoresWIK,comminutyWIK, trainWIK)
testData2 = make_list(sentencesWIK,scoresWIK,comminutyWIK, testWIK)

# write to csv file
write_csv(trainData1+trainData2,'train.csv')
write_csv(testData1+testData2,'test.csv')