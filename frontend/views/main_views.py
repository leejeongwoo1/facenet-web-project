from flask import Blueprint, url_for, render_template, request
from werkzeug.utils import redirect, secure_filename
import PIL
from mtcnn import mtcnn
import numpy as np
import cv2
import os
import json
import sys
import keras
sys.path.append('../../')
from src import embeddings
bp = Blueprint('main',__name__,url_prefix='/')

# def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):
#     try:
#         n = np.fromfile(filename, dtype)
#         img = cv2.imdecode(n, flags)
#         return img
#     except Exception as e:
#         print(e)
#         return None
#
# def imwrite(filename, img, params=None):
#     try:
#         ext = os.path.splitext(filename)[1]
#         result, n = cv2.imencode(ext, img, params)
#
#         if result:
#             with open(filename, mode='w+b') as f:
#                 n.tofile(f)
#             return True
#         else:
#             return False
#     except Exception as e:
#         print(e)
#         return False

@bp.route('/')
def main():
    return render_template('main/main.html',client_img=False,faculty=False)

@bp.route('/',methods=('POST',))
def create():
    model = keras.models.load_model('./model/facenet_keras.h5')
    file = request.files['file']
    path = "./client_img/"+(file.filename)
    file.save(path)
    model = keras.models.load_model('./model/facenet_keras.h5')
    emb = embeddings.get_embedding_from_one_pic(model, path)
    faculty_json = open('./model/faculty_emb.json',encoding='utf-8')
    faculty_dict = json.load(faculty_json)
    min_dist = 100
    name = ""
    for key in faculty_dict:
        dist = np.linalg.norm(faculty_dict[key]-emb)
        if min_dist>dist:
            min_dist=dist
            name = key
    #print(name, min_dist)
    department = name.split('_')[0]
    name_ = name.split('_')[1]
    faculty_path="../../../dataset/faculty/origin_img/"+name+".jpg"
    return render_template('main/main.html', client_img=path, faculty=faculty_path, department=department, name=name_)
