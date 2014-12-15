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
from settings import *
if USE_CLASSIFIER == 'basic_model':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../basic_model'))
    from baselineExtended import compute_score
elif USE_CLASSIFIER == 'baseline':
    sys.path.append(os.path.join(os.path.dirname(__file__), '../baseline'))
    import baseline
    classifier = joblib.load(os.path.join(os.path.dirname(__file__), '../../results/models/classifier.p')) 
    extraction = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../results/models/extraction.p'),'rb'))
if USE_DOP:
    from treeToFeatures import *
from wordCounts import *


if USE_BLLIP == True:
    if BLLIP_TYPE == 'Python':
        from bllipparser import RerankingParser, tokenize
        from get_trees import split_sentences
    else:
        from get_trees import *

topicmodel = None
baselineModel = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../results/models/baseline0.5'+'.p'),'rb'))
#wordScores = pickle.load(open(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/wordScores.p'),'rb'))

# Global variables used in the application
featureMap = {}

def getClassFromNumber(label):
    wordClass = 'neutral'
    if label == 1.0:
        wordClass = 'polite'
    if label == 0.0:
        wordClass = 'impolite'
    return wordClass

def loadDopFeatures():
    global featureMap
    with open(os.path.join(os.path.dirname(__file__), '../../datasets/preprocessed/featureSpace.txt')) as f:
        for line in f:
            i, tree = line.split(' ',1)
            tree = tree.strip()
            featureMap[tree] = i

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")
class PolitenessHandler(tornado.web.RequestHandler):
    def post(self):
        global USE_BLLIP, USE_DOP, BLLIP_TYPE, USE_CLASSIFIER, rrp, featureMap, topicmodel, classifier, extraction #, baselineModel, wordScores
        sentence = self.get_argument("sentence", None)
        if USE_BLLIP == True:
            if BLLIP_TYPE == 'Python':
                tree = [rrp.simple_parse(str(sent)) for sent in split_sentences(sentence)]
            else:
                tree = get_trees(sentence)
        else:
            example_trees = [ '(S1 (S (VP (VB Thank) (NP (PRP you)) (PP (IN for) (NP (DT the) (FW response.) (SQ (MD Would) (NP (PRP you)) (VP (AUX be) (ADJP (JJ willing) (S (VP (TO to) (VP (VB add) (NP (DT a) (JJ few) (JJR more) (NNS details)) (S (VP (TO to) (VP (VB explain) (ADVP (RBR further)))))))))))))) (. ?)))',
            '(S1 (SBARQ (WHNP (WDT That)) (SQ (AUX \'s) (NP (NP (DT the) (JJ only) (NN answer)) (SBAR (S (NP (PRP you)) (VP (AUX have)))) (. ?)) (ADVP (RB seriously)) (. ?) (SQ (MD can) (RB n\'t) (NP (PRP you)) (VP (AUX do) (ADVP (RB better))))) (. ?)))' ]
            tree = np.random.choice(example_trees, 1)
            print sentence
            sentence = sentence
        print "Test"
        print list(tree)
        features = ""
        if USE_DOP:
            features = convertTreeToVector(list(tree), featureMap)
            print features
        # Classifier
        label = 0
        scoreFound = None
        if USE_CLASSIFIER == 'basic_model':
            scoreFound = compute_score(sentence,baselineModel)
            if scoreFound is None:
                scoreFound = 0
            classFound = 'neutral'
            label = 0
            if scoreFound >= 0.5: #-0.38765975611068004, 0.4548283398341303]
                classFound = 'polite'
                label = 1
            if scoreFound <= -0.5:
                classFound = 'impolite'
                label = 0
        elif USE_CLASSIFIER == 'baseline':
            # Baseline
            featuresBaseline = baseline.getFeaturesForSentence(sentence, extraction, False)
            label = classifier.predict(featuresBaseline)[0]
            print label
            if label == 2:
                label = 1
            classFound = getClassFromNumber(label)
        elif USE_CLASSIFIER == 'dopclassifier':
            # Dop feature
            print ""

        sentence = sentence+str(features)
        sentenceWords = list(enumerate([word for word in re.findall(r"[\w']+|[\W]",sentence) if not word==" "]))

        tags = [2]*len(sentenceWords)

        if topicmodel != None:
            print "?"
            result = topicmodel.wordTagsForSentence(sentence, label, num_its = 2)
            print result
            tags[:len(result)] = result

        response = { 'sentences': [ {'sentence': " ".join(["<span class=\""+getClassFromNumber(tags[i])+"\">"+word+" "+"</span>" for (i, word) in sentenceWords if word != "" ]), 'sentenceClass': classFound, 'confidence': scoreFound } ]}
        self.write(response)
        topicmode_trainer().start()

application = tornado.web.Application([
    (r"/politeness/", MainHandler),
    (r"/politeness/classify/", PolitenessHandler)],
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    xsrf_cookies=True,
    debug=True,
    static_url_prefix="/politeness/static/"

)

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
        f = open(os.path.join(os.path.dirname(__file__), '../../results/models/topicModel10.txt'), 'r+b')
        topicmodel = pickle.load(f)
        f.close()
        print "Topic model end"

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

loadDopFeatures()
if __name__ == "__main__":
    application.listen(8888)
    if USE_BLLIP == True:
        if BLLIP_TYPE == 'Python':
            bllip_loader().start()
    topicmode_loader().start()
    tornado.ioloop.IOLoop.instance().start()
