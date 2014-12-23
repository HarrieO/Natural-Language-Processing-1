import tornado.ioloop
import tornado.web
import os.path
import sys, re
import time
import cPickle as pickle
import numpy as np
import sklearn
from sklearn.externals import joblib
from threading import Thread
from tornado.options import define, options, parse_command_line
sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../discofeatures'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../topicmodel'))
from wordCounts import *
from settings import *

# Based on the setttings we load the apropiate files for each of the models
if USE_CLASSIFIER == 'basic_model':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../basic_model'))
    from baselineExtended import compute_score
elif USE_CLASSIFIER == 'baseline':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
    import baseline
    classifier = joblib.load(os.path.join(os.path.dirname(__file__), '../../results/models/baseline_linear_classifier.p')) 
    extraction = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../results/models/last_extraction.p'),'rb'))
if USE_DOP:
    from treeToFeatures import *

# If the BLLIP parser is ued include the bllip parser
if USE_BLLIP == True:
    # There are two ways to use the bllip parser, the python version performs better
    if BLLIP_TYPE == 'Python':
        from bllipparser import RerankingParser, tokenize
        from get_trees import split_sentences
    else:
        from get_trees import *

def getClassFromNumber(label):
    wordClass = 'neutral'
    if label == 1.0:
        wordClass = 'polite'
    if label == 0.0:
        wordClass = 'impolite'
    return wordClass


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

class PolitenessHandler(tornado.web.RequestHandler):
    def post(self):
        global USE_BLLIP, USE_DOP, BLLIP_TYPE, USE_CLASSIFIER, rrp, featureMap, topicmodel, classifier, extraction, patternRemove
        label = 0
        scoreFound = None
        tree = None
        sentence = self.get_argument("sentence", None)
        features = ""

        if USE_BLLIP == True:
            if BLLIP_TYPE == 'Python':
                tree = [rrp.simple_parse(str(sent)) for sent in split_sentences(sentence)]
            else:
                tree = get_trees(sentence)
        else:
            sentence = sentence

        if USE_DOP:
            features = convertTreeToVector(list(tree), featureMap)

        # Classifier
        if USE_CLASSIFIER == 'basic_model':
            scoreFound = compute_score(sentence,baselineModel)
            if scoreFound is None:
                scoreFound = 0
            classFound = 'neutral'
            label = 0
            if scoreFound >= 0.5:
                classFound = 'polite'
                label = 1
            if scoreFound <= -0.5:
                classFound = 'impolite'
                label = 0
        elif USE_CLASSIFIER == 'baseline':
            # Baseline
            featuresBaseline = baseline.getFeaturesForSentence(sentence, extraction, False)
            print featuresBaseline
            label = classifier.predict(featuresBaseline)[0]
            print label
            if label == 2:
                label = 1
            elif label == 1:
                label = 2
            classFound = getClassFromNumber(label)
        elif USE_CLASSIFIER == 'dopclassifier':
            # Dop feature
            # Not implemented in the demo
            print ""


        sentenceWords = list(enumerate([word for word in re.findall(r"[\w']+|[\W]",sentence) if not word==" "]))

        tags = [2]*len(sentenceWords)
        if topicmodel != None and label != 2:
            result = topicmodel.wordTagsForSentence(sentence, label)
            sentence = sentence+str(features)
            n = 0
            for (i, word) in sentenceWords:
                word = patternRemove.sub('', str(word)).lower()
                if len(word) > 0 and n<len(result):
                    tags[i] = result[n]
                    n += 1

        response = { 'sentences': [ {'sentence': " ".join(["<span class=\""+getClassFromNumber(tags[i])+"\">"+word+" "+"</span>" for (i, word) in sentenceWords if word != "" ]), 'sentenceClass': classFound, 'confidence': scoreFound } ]}
        self.write(response)


class bllip_loader(Thread):
    def run(self):
        global rrp
        print "Reloading"
        rrp = RerankingParser()
        rrp.load_parser_model(os.path.join(os.path.dirname(__file__), '../../lib/bllip/DATA/EN'))
        rrp.load_reranker_model(os.path.join(os.path.dirname(__file__), '../../lib/bllip/models/ec50spfinal/features.gz'), os.path.join(os.path.dirname(__file__), '../../lib/bllip/models/ec50spfinal/cvlm-l1c10P1-weights.gz'))
        print "Done loading model"


class topicmode_loader(Thread):
    def run(self):
        global topicmodel
        print "Topic model"
        f = open(os.path.join(os.path.dirname(__file__), '../../results/models/topicModel200000.txt'), 'r+b')
        topicmodel = pickle.load(f)
        f.close()
        print "Topic model end"

class dop_features_loader(Thread):
    def run(self):
        global featureMap
        with open(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/featureSpace.txt')) as f:
            for line in f:
                i, tree = line.split(' ',1)
                tree = tree.strip()
                featureMap[tree] = i

class topicmode_trainer(Thread):
    def run(self):
        global topicmodel
        if topicmodel != None:
            num_its = 10
            n = len(topicmodel.sentenceTags)-1
            numWords = len(topicmodel.sentenceTags[n])-1
            for i in range(num_its):
                m = random.randrange(numWords)
                topicmodel.conditional_distribution((n,m))

# Global variables used in the application
featureMap = {}
patternRemove = re.compile('[\W_]+', re.UNICODE)
topicmodel = None
baselineModel = None

application = tornado.web.Application([
    (r"/politeness/", MainHandler),
    (r"/politeness/classify/", PolitenessHandler)],
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    xsrf_cookies=True,
    debug=True,
    static_url_prefix="/politeness/static/"

)

if __name__ == "__main__":
    # Here we start the web server, and in seperate threads load data and models
    application.listen(8888)
    if USE_CLASSIFIER == 'basic_model':
        baselineModel = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../results/models/baseline0.5'+'.p'),'rb'))
    if USE_BLLIP == True:
        if BLLIP_TYPE == 'Python':
            bllip_loader().start()
    if USE_DOP:
        dop_features_loader().start()
    topicmode_loader().start()
    tornado.ioloop.IOLoop.instance().start()
