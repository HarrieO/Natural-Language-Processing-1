#!/usr/bin/env python2
# coding=utf-8
from datapoint import *

class FeatureDeduction(object):
    def __init__(self):
        self.features = []
        self.entropy  = []
        with open('../../datasets/preprocessed/informationGain.txt') as f:
            for line in f:
                fid, entropy = line.split(',',1)
                self.features.append(int(fid))
                self.entropy.append(float(entropy.strip()))

        self.trees = [None]*len(self.features)
        with open('../../datasets/preprocessed/featureSpace.txt') as f:
            for line in f:
                i, tree = line.split(' ',1)
                tree = tree.strip()
                self.trees[int(i)] = tree

        self.avescores = [0]*len(self.features)
        self.total     = [0]*len(self.features)
        self.polite    = [0]*len(self.features)
        self.impolite  = [0]*len(self.features)
        with open('../../datasets/preprocessed/indicesToAverageScore.txt') as f:
            for line in f:
                i, score, total, polite, impolite = line.split()
                i = int(i)
                self.avescores[i] = float(score)
                self.total[i]     = int(total)
                self.polite[i]    = int(polite)
                self.impolite[i]  = int(impolite)
        posts = read_data('../../datasets/preprocessed/trainset.csv')
        self.all_total    = len(posts)
        self.all_polite   = 0
        self.all_impolite = 0
        for post in posts:
            if post.score > 0.5:
                self.all_polite += 1
            elif post.score < -0.5:
                self.all_impolite += 1

    def printTrees(self,indices):
        for i in indices:
            self.printTree(i)

    def printTree(self,tree_index):
        print "Feature_id:", tree_index, "DOP_id:", self.features[tree_index]
        print "Information Gain:", abs(self.entropy[tree_index])
        tree_index = self.features[tree_index]
        print "Average Score:", self.avescores[tree_index]
        print "Precision total:", self.total[tree_index], " polite:", round(self.polite[tree_index]/float(self.total[tree_index]),2), 
        print " impolite:", round(self.impolite[tree_index]/float(self.total[tree_index]),2),
        print " neutral:", round((self.total[tree_index] - self.polite[tree_index] - self.impolite[tree_index])/float(self.total[tree_index]),2)
        print "Recall Total", round(float(self.total[tree_index])/self.all_total,2),
        print " polite", round(float(self.polite[tree_index])/self.all_polite,2),
        print " impolite", round(float(self.impolite[tree_index])/self.all_impolite,2),
        print " neutral", round((self.total[tree_index] - self.impolite[tree_index] - self.polite[tree_index])/float(self.all_total - self.all_impolite - self.all_polite),2)
        print self.trees[tree_index]
        print


f = FeatureDeduction()
f.printTrees(range(25))

