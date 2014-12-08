import numpy as np

# returns the features for a given list of (maybe unseen) words, applying "smoothing" to unseen words
def extract_features_word(words,leaveOutWords, featureWords):
	# words = list of words
	# zeroScoreList = list of words that are left out
	# featureWords = list of words that are seen as features 
	features = dict()
	# initialize counts
	for word in featureWords:
		features[word]=0
	for word in words:
		if word in leaveOutWords:
			continue
		# count seen words
		if word in featureWords:
			features[word]+=1
		# else:
		# 	wordList = list(word)
		# 	# find largest last part that is the same a word that is already seen
		# 	for i in range(len(wordList)+1):
		# 		sequence = wordList[i:]
		# 		number, positives = contains_end(sequence,leaveOutWords,featureWords)
		# 		if len(positives) >0:
		# 			for entry in positives:
		# 				features[entry] += 1.0/float(number)
		# 			break
	return features

def contains_end(sequence,ignoreWords, addWords):
	length = len(sequence)
	positives = []
	for word in addWords:
		wList = list(word)
		if len(wList)<length:
			continue
		if wList[-len(sequence):]== sequence:
			positives.append(word)
	number = len(positives)
	for word in ignoreWords:
		wList = list(word)
		if len(wList)<length:
			continue
		if wList[-len(sequence):]== sequence:
			number +=1
	return number,positives


if __name__ == "__main__":
	print extract_features_word(['hi','there'], ['hi','are','bare'],['snare', 'hi'])
	print extract_features_word(['(VB Thank)','(PRP you)'], [],['(DT the)', '(VB Thank)'])