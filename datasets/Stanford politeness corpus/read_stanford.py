import csv
import sys

maxInt = int(sys.maxsize/(10**10))
csv.field_size_limit(maxInt)

with open('stack-exchange.annotated.csv') as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	rowCounter = 0
	for row in spamreader:
		rowCounter+1
		counter = 0
		for col in row:
			if counter == 0:
				print col
				break
			counter = counter+1
	print rowCounter


