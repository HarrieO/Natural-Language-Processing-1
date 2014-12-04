import sys, os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../disco')
#sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))

import treepost

fd = open('classifier_results.csv','r')

