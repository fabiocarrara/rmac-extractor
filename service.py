import os
import time
import cPickle
import datetime
import logging
import flask
import werkzeug
import argparse
import tornado.wsgi
import tornado.httpserver
import numpy as np
from PIL import Image
import cStringIO as StringIO
import urllib

from extractor import RMACExtractor

ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif'])

def embed_image_html(image):
    """Creates an image embedded in HTML base64 format."""
    image_pil = Image.fromarray((255 * image).astype('uint8'))
    image_pil = image_pil.resize((256, 256))
    string_buf = StringIO.StringIO()
    image_pil.save(string_buf, format='png')
    data = string_buf.getvalue().encode('base64').replace('\n', '')
    return 'data:image/png;base64,' + data


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

# Obtain the flask app object
app = flask.Flask(__name__)


@app.route('/extract', methods=['POST'])
def extract_from_image():
    """ Extracts RMAC features given an uploaded image.
        TODO: test features, expose multiScale, S and L
    """
    try:
        image = Image.open(flask.request.files['image'].stream)
        logging.info('Image: %s', image)
    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return str(err), 400

    S = flask.request.values.getlist('s', int) if 's' in flask.request.values else (500,)
    features = np.stack([app.extractor.extract_from_pil(image, S=s) for s in S]).sum(axis=0)
    features = features / np.linalg.norm(features)
    return flask.jsonify(features.tolist())


@app.route('/extract', methods=['GET'])
def extract_from_url():
    """ Extracts RMAC features given an image URL.
        TODO: test features, expose multiScale, S and L
    """
    image_url = flask.request.args.get('url', '')
    logging.info('Image: %s', image_url)
    try:
        string_buffer = StringIO.StringIO(urllib.urlopen(image_url).read())
        image = Image.open(string_buffer)

    except Exception as err:
        # For any exception we encounter in reading the image, we will just
        # not continue.
        logging.info('URL Image open error: %s', err)
        return str(err), 400

    S = flask.request.args.getlist('s', int) if 's' in flask.request.args else (500,)
    features = np.stack([app.extractor.extract_from_pil(image, S=s) for s in S]).sum(axis=0)
    features = features / np.linalg.norm(features)
    return flask.jsonify(features.tolist())


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


def start_from_terminal(app):
    """
    Parse command line options and start the server.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', help="enable debug mode", action="store_true", default=False)
    parser.add_argument('-p', '--port', help="which port to serve content on", type=int, default=5000)
    parser.add_argument('-g', '--gpu', help="which gpu to use (-1 for cpu)", type=int, default=-1)

    args = parser.parse_args()
    app.extractor = RMACExtractor(args.gpu)

    if args.debug:
        app.run(debug=True, host='0.0.0.0', port=args.port)
    else:
        start_tornado(app, args.port)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    start_from_terminal(app)
