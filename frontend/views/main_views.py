from flask import Blueprint, render_template, request
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
import sys
from PIL import Image
import time
sys.path.append('../../')
bp = Blueprint('main',__name__,url_prefix='/')

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

@bp.route('/')
def main():
    return render_template('main/main.html',client_img=False,faculty=False)

@bp.route('/',methods=('POST',))
def create():
    start=time.time()

    file = request.files['file']
    path = "./frontend/static/client_img/"+(file.filename)
    path_ = "./static/client_img/"+(file.filename)
    file.save(path)
    img = Image.open(path)
    img_cropped = mtcnn(img)

    emb = resnet(img_cropped.unsqueeze(0))[0].detach().numpy()
    faculty_json = open('./model/faculty_emb.json',encoding='utf-8')
    faculty_dict = json.load(faculty_json)
    min_dist = 100
    name = ""
    for key in faculty_dict:
        print('faculty:',type(faculty_dict[key][0]))
        print('emb:',type(emb[0]))
        dist = np.linalg.norm(faculty_dict[key]-emb)
        if min_dist>dist:
            min_dist=dist
            name = key
    print(name, min_dist)
    department = name.split('_')[0]
    name_ = name.split('_')[1]
    faculty_path = "./static/faculty_img/"+name+".jpg"

    end=time.time()
    print(f"전체 소요 시간 : {end-start:.5f}")
    return render_template('main/main.html', client_img=path_, faculty=faculty_path, department=department, name=name_)