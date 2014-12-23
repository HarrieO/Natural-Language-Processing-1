import post
import numpy as np
import pylab as P

contents = map(float,post.read_column(1,'../datasets/preprocessed/train.csv'))
n, bins, patches = P.hist(contents, 50, histtype='stepfilled')
P.axvline(x=-0.38765975611068004, linewidth=3, color='r')
P.axvline(x=0.4548283398341303, linewidth=3, color='r')
P.figure()
P.show()