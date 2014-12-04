import tornado.ioloop
import tornado.web
import os.path
import sys
import time
from threading import Thread
from tornado.options import define, options, parse_command_line
sys.path.append(os.path.join(os.path.dirname(__file__), '../disco'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from settings import *


if USE_BLLIP == True:
	if BLLIP_TYPE == 'Python':
		from bllipparser import RerankingParser, tokenize
	else:
		from get_trees import *


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")
class PolitenessHandler(tornado.web.RequestHandler):
    def post(self):
    	global USE_BLLIP, BLLIP_TYPE, rrp
    	sentence = self.get_argument("sentence", None)
    	if USE_BLLIP == True:
		if BLLIP_TYPE == 'Python':
			sentence = rrp.simple_parse(str(sentence))
		else:
    			sentence = get_trees(sentence)
    			sentence = sentence[0]
		# Test trees (S1 (S (VP (VB Thank) (NP (PRP you)) (PP (IN for) (NP (DT the) (FW response.) (SQ (MD Would) (NP (PRP you)) (VP (AUX be) (ADJP (JJ willing) (S (VP (TO to) (VP (VB add) (NP (DT a) (JJ few) (JJR more) (NNS details)) (S (VP (TO to) (VP (VB explain) (ADVP (RBR further)))))))))))))) (. ?)))
		# (S1 (SBARQ (WHNP (WDT That)) (SQ (AUX 's) (NP (NP (DT the) (JJ only) (NN answer)) (SBAR (S (NP (PRP you)) (VP (AUX have)))) (. ?)) (ADVP (RB seriously)) (. ?) (SQ (MD can) (RB n't) (NP (PRP you)) (VP (AUX do) (ADVP (RB better))))) (. ?)))
    	else:
    		sentence = sentence
        response = { 'sentences': [ {'sentence': " ".join(["<span class=\"neutral\">"+word+"</span>" for word in sentence.split(" ") if word != "" ]), 'sentenceClass': 'polite' } ]}
        self.write(response)

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


if __name__ == "__main__":
	application.listen(8888)
	if USE_BLLIP == True:
		if BLLIP_TYPE == 'Python':
			bllip_loader().start()
	tornado.ioloop.IOLoop.instance().start()
