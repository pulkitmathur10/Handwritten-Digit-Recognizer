# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 23:08:46 2019

@author: Pulkit
"""

from flask import Flask, request, render_template, jsonify
from cnn_model import cnn_predict
import numpy as np
from PIL import Image
import re
import io
import base64

app = Flask(__name__)


@app.route("/",methods=["POST", "GET"])
def output():
    y=0
    if request.method== 'POST':
        url = request.values['imageBase64'] 
        image_string = re.search(r'base64,(.*)', url).group(1)  
        image_bytes = io.BytesIO(base64.b64decode(image_string)) 
        image = Image.open(image_bytes) 
        image = image.resize((28, 28), Image.ANTIALIAS)  
        image = image.convert('LA') #RGB to Black & White
        image_to_array = np.asarray(image)
        image_to_array = image_to_array[:, :, 1]
        image_to_array = image_to_array.astype('float32') / 255
        image_to_array = image_to_array.reshape((1, 28, 28, 1))
        y = cnn_predict(image_to_array)
        return jsonify(y = y)
    return render_template("inp1.html",y=y)


if __name__ == "__main__":
    app.run()
