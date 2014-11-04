import csv
import sys
import numpy as np

maxInt = int(sys.maxsize/(10**10))
csv.field_size_limit(maxInt)

def read_column(colNr):
	with open('stack-exchange.annotated.csv') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=',')
		row = next(spamreader)
		column = []
		for row in spamreader:
			counter = 0
			for col in row:
				if counter == colNr:
					column.append(col)
					break
				counter = counter+1
		return column

col1 = np.array(read_column(2)) # sentences
col2 = np.array(read_column(13)) # mean score
indices = np.array(np.argsort(col2))
print col2[indices]
print col1[indices]