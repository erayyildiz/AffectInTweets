# -*- coding: utf-8 -*-
import tornado.web
import os
from tornado import web
from affect_predictor import AffectPredictor


class StaticGlovbalObjects:
    AFFECT_PREDICTOR = AffectPredictor()

    def __init__(self):
        pass


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("template.html")


class TweetHandler(tornado.web.RequestHandler):
    CSS_CLASS_DIC = {"Duygu yok": "no-emo", "Düşük": "low-emo", "Orta": "mid-emo", "Yüksek": "high-emo"}

    def get(self):
        tweet = self.get_argument("tweet")
        results = StaticGlovbalObjects.AFFECT_PREDICTOR.predict(tweet)
        self.render("template.html", results=results,
                    class_dic=TweetHandler.CSS_CLASS_DIC,
                    tweet=tweet)


def make_app():
    public_root = os.path.dirname(__file__)
    settings = dict(
        debug=True,
        static_path=public_root,
        template_path=public_root
    )
    handlers = [
        (r"/", MainHandler),
        (r"/measure", TweetHandler),
        (r'/(.*)', web.StaticFileHandler, {'path': public_root}),
    ]
    return web.Application(handlers, **settings)

if __name__ == "__main__":
    app = make_app()
    app.listen(8891)
    tornado.ioloop.IOLoop.current().start()