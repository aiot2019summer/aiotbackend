#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response
from flask_caching import Cache
import webbrowser


# import camera driver
if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from camera import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

cache=Cache()
app = Flask(__name__)
cache.init_app(app)


@app.route('/')
@cache.cached(timeout=300,key_prefix='index')
def index():
    """Video streaming home page."""
    return render_template('index.html')


def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    cache.clear()
    return Response(gen(Camera()),
        mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    #webbrowser.open("http://localhost/templates/index.html")
    app.run(host='0.0.0.0', threaded=True)
