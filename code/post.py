import csv
import sys
import numpy as np

class Post(object):
	def __init__(self, content, score, community):
		self.content = " ".join(content.split())
		self.score = score
		self.community = community


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

if __name__ == '__main__':
	contents = read_column(0,'train.csv')
	scores = read_column(1,'train.csv')

def read_posts(fileName):
	with open(fileName) as csvfile:
		readerObject = csv.reader(csvfile, delimiter=',')
		# skip header of csv file
		row = next(readerObject)
		posts = []
		for row in readerObject:
			posts.append(Post(*row[:3]))
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
	contents    = read_column(0,'train.csv')
	scores      = read_column(1,'train.csv')
	communities = read_column(2,'train.csv')
	print contents[0:2]
	print scores[0:2]
	print communities[0:2]

	for post in read_posts('train.csv'):
		print post.content, post.score, post.community

	print read_table('train.csv')

