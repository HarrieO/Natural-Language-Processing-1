import csv
import sys
import numpy as np
import ast

class DataPoint(object):
	def __init__(self, tid, content, score, community, fragments):
		self.id        = int(tid)
		self.content   = " ".join(content.split())
		self.score     = float(score)
		self.community = community
		frag = ast.literal_eval(fragments)
		self.fragments = {}
		for key, elem in frag.items():
			self.fragments[int(key)] = float(elem)


maxInt = int(2**31-1)
csv.field_size_limit(maxInt)

def read_column(colNr, fileName):
	with open(fileName) as csvfile:
		readerObject = csv.reader(csvfile, delimiter=',')
		row = next(readerObject)
		column = []
		for row in readerObject:
			column.append(row[colNr])
		return column

def read_data(fileName):
	with open(fileName) as csvfile:
		readerObject = csv.reader(csvfile, delimiter=',')
		# skip header of csv file
		row = next(readerObject)
		posts = []
		for row in readerObject:
			posts.append(DataPoint(*row))
		return posts

def read_table(fileName):
	with open(fileName) as csvfile:
		readerObject = csv.reader(csvfile, delimiter=',')
		# skip header of csv file
		row = next(readerObject)
		table = [] 
		for row in readerObject:
			table.append(row)
		return table

if __name__ == "__main__":
	for post in read_posts('../../datasets/preprocessed/featureData.csv'):
		print post.id, post.fragments


