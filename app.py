from fastbook import *
import fastai
from util import base64_to_pil

import os
import numpy as np
# https://stackoverflow.com/questions/31684375/automatically-create-file-requirements-txt


# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

##### load model
path = Path()
path.ls(file_exts='.pkl')
learn_inf = load_learner(path/'export.pkl')


app = Flask(__name__)

def get_ImageClassifierModel():
    model = load_learner('export.pkl')
    return model


def model_predict(img, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    # img = img.resize((128, 128))
    # img = fastai.vision.core.PILImage.create(np.array(img.convert('RGB')))
    pred,pred_idx,probs = learn_inf.predict(img)
    
    return pred, pred_idx, probs


@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        img = img.resize((128, 128))
        img = fastai.vision.core.PILImage.create(np.array(img.convert('RGB')))

        # initialize model
        model = get_ImageClassifierModel()

        # Make prediction
        pred, pred_idx, probs = model_predict(img, model)

        # pred_proba = "{:.3f}".format(np.amax(probs))    # Max probability
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        # Serialize the result, you can add additional fields
        return jsonify(result=pred, probability=str(round(probs[pred_idx].item(), 2)))
    return None


if __name__ == '__main__':
    # app.run(port=5002)
    app.run()
