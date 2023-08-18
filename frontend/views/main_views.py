from flask import Blueprint, render_template, request
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
import sys
from PIL import Image
sys.path.append('../../')
bp = Blueprint('main',__name__,url_prefix='/')

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

@bp.route('/')
def main():
    return render_template('main/main.html',client_img=False,faculty=False)

@bp.route('/',methods=('POST',))
def create():

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
        #print('faculty:',type(faculty_dict[key][0]))
        #print('emb:',type(emb[0]))
        normalized_faculty_emb = faculty_dict[key]/np.linalg.norm(faculty_dict[key])
        normalized_emb = emb/np.linalg.norm(emb)
        cosine_sim = np.dot(normalized_emb,normalized_faculty_emb)
        dist = abs(1-cosine_sim)
        if min_dist>dist:
            min_dist=dist
            name = key
    print(name, min_dist)
    department = name.split('_')[0]
    name_ = name.split('_')[1]
    faculty_path = "./static/faculty_img/"+name+".jpg"

    return render_template('main/main.html', client_img=path_, faculty=faculty_path, department=department, name=name_)
