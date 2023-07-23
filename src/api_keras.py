import joblib
import os
import numpy as np
from numpy.lib.shape_base import expand_dims
from embeddings import extract_face, get_embedding    # 얼굴영역 추출, embedding vector로 변환
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont     #이미지 크기 변경
from mtcnn import mtcnn    #얼굴 탐색
from sklearn import preprocessing    #전처리
from flask import Flask, jsonify, request
from waitress import serve
import io
import base64

model = keras.models.load_model('../model/facenet_keras.h5')    # FaceNet model
classifier = joblib.load('../model/class_classifier.pkl')    # 학습된 SVM classifier      
out_encoder = joblib.load('../model/class_labeler.pkl')    # classifier가 예측한 원핫벡터를 문자열 label로 변화하는 LabelEncoder
app = Flask(__name__)

def make_prediction(input):
    pass

@app.route('/lovelyz', methods=['POST'])    #api의 url : /lovelyz
def predict():
    if request.method == 'POST':
        file = request.files.get('file', '')
        img_bytes = file.read()    
        face = extract_face(io.BytesIO(img_bytes))    # 1. 얼굴 영역 추출 후 원하는 size로 이미지 return 
        embedding = get_embedding(model, face)    # 2. FaceNet model설정에 맞게 embedding vector로 변환
        embedding = expand_dims(embedding, axis=0)   #get_embedding에서 이미 하지 않았나?

        # 임베딩 벡터를 단위 벡터로 변환
        in_encoder = preprocessing.Normalizer(norm='l2')    
        embedding = in_encoder.transform(embedding)

        yhat = classifier.predict(embedding)    
        prob = classifier.predict_proba(embedding)    # 예측 확률
        label = out_encoder.inverse_transform(yhat)    # 예측 결과 label
        
        # 이미지에서 얼굴 영역 추출(RGB convert?)
        img = Image.open(io.BytesIO(img_bytes))
        det = mtcnn.MTCNN()
        results = det.detect_faces(np.asarray(img))
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        # 추출한 얼굴영역 box draw
        draw = ImageDraw.Draw(img)
        draw.rectangle(((x1,y1), (x1+width, y1+height)), outline=(255,0,0), width=3)
        draw.rectangle(((x1,y1-height//10),(x1+width//2, y1)), fill=(255,0,0))
        font = ImageFont.truetype('arial.ttf', height//10)
        draw.text((x1, y1-height//10), label[0], fill=(255,255,255), font=font)

        # draw한 이미지 저장(ret)
        ret_bytes = io.BytesIO()
        img.save(ret_bytes, format="PNG")
        ret = base64.encodebytes(ret_bytes.getvalue()).decode('ascii')

        # 예측 정답, 예측 확률, 얼굴 추출된 입력 이미지 output 
        output = dict()
        output['label'] = label[0]
        output['prob'] = round(prob[0][yhat][0]*100, 3)
        output['retImg'] = ret
        
        response = jsonify(output)
        response.headers.add('Access-Control-Allow-Origin', '*')

        return response

if __name__ == '__main__':
    serve(app, port=5000)