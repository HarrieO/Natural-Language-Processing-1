import csv
import sys
import numpy as np

class Post(object):
	def __init__(self, content, score, community):
		self.content = content
		self.score = score
		self.community = community


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

if __name__ == '__main__':
	contents = read_column(0,'train.csv')
	scores = read_column(1,'train.csv')
	communities = read_column(2,'train.csv')
	print contents[0:2]
	print scores[0:2]
	print communities[0:2]
