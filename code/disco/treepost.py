import csv
import sys
import numpy as np

class Post(object):
	def __init__(self, tid, content, score, community, trees):
		self.id        = tid
		self.content   = " ".join(content.split())
		self.score     = score
		self.community = community
		self.trees     = self.parseTrees(trees)

	def parseTrees(self, treesString):
		start = treesString.find("<tree>")
		end   = treesString.find("</tree>")
		trees = []
		while start >= 0:
			trees += [treesString[start+6:end]]
			start = treesString.find("<tree>", end)
			end   = treesString.find("</tree>", end+1)
		return trees

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

def read_posts(fileName):
	with open(fileName) as csvfile:
		readerObject = csv.reader(csvfile, delimiter=',')
		# skip header of csv file
		row = next(readerObject)
		posts = []
		for row in readerObject:
			posts.append(Post(*row))
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
	for post in read_posts('discotrain.csv'):
		print post.id, post.trees


