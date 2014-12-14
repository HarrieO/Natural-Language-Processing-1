#!/usr/bin/env python2
# coding=utf-8
from discodop import treebank, treetransforms, fragments
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))
from bracketStringReader import BracketStringReader


# takes the trees of a single post and returns a sparse feature vector
def convertTreeToVector(treeStringsInput, featureMap):
    totalTrees = len(treeStringsInput)
    treeStrings = treeStringsInput[:]
    treeStrings.extend(treeStrings[:])

    text = BracketStringReader(treeStrings)

    trees = [treetransforms.binarize(tree, horzmarkov=1, vertmarkov=1)
             for _, (tree, _) in text.itertrees(0)]

    sents = [sent for _, (_, sent) in text.itertrees(0)]

    result = fragments.getfragments(trees, sents, numproc=1, disc=False, cover=True)

    featureVector = {} 
    found = 0
    total = 0
    for tree, sentDict in result.items():
        total += 1
        if tree in featureMap:
            found += 1
             #print '%3d\t%s' % (sum(b.values()), a)
            for key, count in sentDict.items():
                if key < totalTrees:
                    treeIndex = featureMap[tree.strip()]
                    if treeIndex in featureVector:
                        featureVector[treeIndex] += count
                    else:
                        featureVector[treeIndex]  = count

    print "Found", found, "out of", total,"trees in featurespace."

    return featureVector



if __name__ == '__main__':
    
    featureMap = {}
    with open('../../datasets/preprocessed/featureSpace.txt') as f:
        for line in f:
            i, tree = line.split(' ',1)
            tree = tree.strip()
            featureMap[tree] = i
    print convertTreeToVector(["(S1 (S (NP (PRP You) (NN error)) (VP (VBZ seems) (S (VP (TO to) (VP (AUX have) (NP (NP (NN something)) (SBAR (S (VP (TO to) (VP (AUX do) (X (TO to)) (PP (IN with) (NP (NP (DT the) (NN function) (POS 's)) (NN invocation)))))))))))) (. .)))", '(S1 (SQ (MD Can) (NP (PRP you)) (VP (VB tell) (NP (PRP us)) (SBAR (WHADVP (WRB where)) (S (VP (AUX is) (NP (NN line) (CD 26)))))) (. ?)))'],featureMap)
