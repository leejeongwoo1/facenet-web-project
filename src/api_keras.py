import joblib
import os
import numpy as np
from numpy.lib.shape_base import expand_dims
from embeddings import extract_face, get_embedding
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont
from mtcnn import mtcnn
from sklearn import preprocessing
from flask import Flask, jsonify, request
from waitress import serve
import io
import base64

model = keras.models.load_model('../model/facenet_keras.h5')
classifier = joblib.load('../model/class_classifier.pkl')
out_encoder = joblib.load('../model/class_labeler.pkl')
app = Flask(__name__)

def make_prediction(input):
    pass

@app.route('/lovelyz', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file', '')
        img_bytes = file.read()
        face = extract_face(io.BytesIO(img_bytes))
        embedding = get_embedding(model, face)
        embedding = expand_dims(embedding, axis=0)

        in_encoder = preprocessing.Normalizer(norm='l2')
        embedding = in_encoder.transform(embedding)

        yhat = classifier.predict(embedding)
        prob = classifier.predict_proba(embedding)
        label = out_encoder.inverse_transform(yhat)

        img = Image.open(io.BytesIO(img_bytes))
        det = mtcnn.MTCNN()
        results = det.detect_faces(np.asarray(img))
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        draw = ImageDraw.Draw(img)
        draw.rectangle(((x1,y1), (x1+width, y1+height)), outline=(255,0,0), width=3)
        draw.rectangle(((x1,y1-height//10),(x1+width//2, y1)), fill=(255,0,0))
        font = ImageFont.truetype('arial.ttf', height//10)
        draw.text((x1, y1-height//10), label[0], fill=(255,255,255), font=font)

        ret_bytes = io.BytesIO()
        img.save(ret_bytes, format="PNG")
        ret = base64.encodebytes(ret_bytes.getvalue()).decode('ascii')

        output = dict()
        output['label'] = label[0]
        output['prob'] = round(prob[0][yhat][0]*100, 3)
        output['retImg'] = ret
        response = jsonify(output)
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response

if __name__ == '__main__':
    serve(app, port=5000)