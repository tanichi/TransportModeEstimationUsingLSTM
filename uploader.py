# -*- coding: utf-8 -*-

import io
import tornado.web
from datetime import datetime 

class PostHandler(tornado.web.RequestHandler):
    def post(self):
        print(datetime.now().strftime("%H:%M:%S"), self.request.files['file'][0]['filename'])
        data = io.BytesIO(self.request.files['file'][0]['body'])
        f = open("upload/" + self.request.files['file'][0]['filename'], 'w')
        
        for line in data:
            f.write(line.decode('utf-8'))

# WebAPIの起動
application = tornado.web.Application([
    (r"/", PostHandler)
])

if __name__ == "__main__":
    port = 8000
    print(port)
    application.listen(port)
    tornado.ioloop.IOLoop.instance().start()
