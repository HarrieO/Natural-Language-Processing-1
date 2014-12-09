#!/usr/bin/env python2
# coding=utf-8


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

    def printTrees(self,indices):
        for i in indices:
            self.printTree(i)

    def printTree(self,tree_index):
        print "Feature_id:", tree_index, "DOP_id:", self.features[tree_index]
        print "Information Gain:", abs(self.entropy[tree_index])
        tree_index = self.features[tree_index]
        print "Average Score:", self.avescores[tree_index]
        print "Total posts:", self.total[tree_index], " polite:", round(self.polite[tree_index]/float(self.total[tree_index]),2), 
        print " impolite:", round(self.impolite[tree_index]/float(self.total[tree_index]),2),
        print " neutral:", round((self.total[tree_index] - self.polite[tree_index] - self.impolite[tree_index])/float(self.total[tree_index]),2)
        print self.trees[tree_index]
        print


f = FeatureDeduction()
f.printTrees(range(100))

