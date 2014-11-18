#!/usr/bin/env python2
# coding=utf-8
import glob, sys
from discodop import treebank, treetransforms, fragments
from sklearn import linear_model, preprocessing, feature_extraction, cross_validation

print 'Hello world'

sys.path.append('..')

import post 
#Per datapunt:
#> Vind elke zin ? . NOT .” ?”
#> Zet elke zin binnen <s> </s> TAGS
#> Elke zin een regel
#> Maak een index

#> Disco dop op file met zinnen (featurespace)

#> Zie tutorial hoe deze features per datapunt
 
#Enters weghalen:
#“ “.join(x.split())

print 'Goodbye world'