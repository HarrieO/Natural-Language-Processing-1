#!/usr/bin/env python2
# coding=utf-8


class FeatureDeduction(object):
    def __init__(self, feature_count):
        self.featureMap = {}
        with open('../../datasets/preprocessed/informationGain.txt') as f:
            for line in f:
                fid, entropy = line.split(',',1)
                self.featureMap[int(fid)] = float(entropy)
                if feature_count:
                    feature_count -= 1
                    if feature_count <= 0:
                        break

    def featureDeduct(self,featureHistogram):
        deductedHistogram = {}
        for fid, elem in featureHistogram.items():
            fid = int(fid)
            if fid in self.featureMap:
                deductedHistogram[fid] = elem
        return deductedHistogram




