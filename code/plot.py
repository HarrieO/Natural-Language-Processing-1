import post
import numpy as np
import pylab as P

contents = map(float,post.read_column(1,'train.csv'))
n, bins, patches = P.hist(contents, 50, histtype='stepfilled')
P.figure()
P.show()