import numpy as np
import re, string, random, time, pickle
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
from datapoint import *
from scipy.misc import logsumexp
pattern = re.compile('[\W_]+', re.UNICODE)
from wordCounts import *

f = open('topicModel10.txt', 'r+')
wordCounter = pickle.load(f)
f.close()

f = open('topicModel10LINUX.txt', 'w+b')
pickle.dump(wordCounter,f)