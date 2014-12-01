import tornado.ioloop
import tornado.web
import os.path
import sys
from tornado.options import define, options, parse_command_line

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")
class PolitenessHandler(tornado.web.RequestHandler):
    def post(self):
    	sentence = self.get_argument("sentence", None)
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

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()