import tornado.ioloop
import tornado.web
import os.path
from tornado.options import define, options, parse_command_line

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")

application = tornado.web.Application([
    (r"/", MainHandler)],
    template_path=os.path.join(os.path.dirname(__file__), "templates"),
    static_path=os.path.join(os.path.dirname(__file__), "static"),
    xsrf_cookies=True
)

if __name__ == "__main__":
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()