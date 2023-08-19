from flask import Flask, Blueprint, render_template, request, flash
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
import sys
import os
from PIL import Image
sys.path.append('../../')
import PIL
bp = Blueprint('main', __name__, url_prefix='/')

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

@bp.route('/')
def main():
    return render_template('main/main.html', client_img=False, faculty=False)

@bp.route('/', methods=('POST',))
def create():
    try:
        folder = "./frontend/static/client_img"
        for i in os.listdir("./frontend/static/client_img"):
            if i != "client_img.txt":
                os.remove(os.path.join(folder,i))

        file = request.files['file']
        path = "./frontend/static/client_img/" + (file.filename)
        path_ = "./static/client_img/" + (file.filename)
        file.save(path)
        img = Image.open(path)
        img_cropped = mtcnn(img)

        emb = resnet(img_cropped.unsqueeze(0))[0].detach().numpy()
        faculty_json = open('./model/faculty_emb.json', encoding='utf-8')
        faculty_dict = json.load(faculty_json)
        min_dist = 100
        name = ""
        for key in faculty_dict:
            print('faculty:', type(faculty_dict[key][0]))
            print('emb:', type(emb[0]))
            dist = np.linalg.norm(faculty_dict[key] - emb)
            if min_dist > dist:
                min_dist = dist
                name = key
        print(name, min_dist)
        similarity_ = ((2-min_dist)/2)*100
        similarity = round(similarity_,2)
        department = name.split('_')[0]
        name_ = name.split('_')[1]
        faculty_path = "./static/faculty_img/" + name + ".jpg"
        return render_template('main/main.html', client_img=path_, faculty=faculty_path, department=department, name=name_,similarity=similarity)
    except (RuntimeError, AttributeError):
        return render_template('main/error_message.html',error_message="얼굴을 인식하지 못했습니다")
    except PIL.UnidentifiedImageError:
        return render_template('main/error_message.html', error_message="올바른 형식이 아닙니다")
    except PermissionError:
        return render_template('main/error_message.html', error_message="입력된 이미지가 없습니다")
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(exc_type, exc_value, exc_traceback)
        return render_template('main/error_message.html',error_message="알 수 없는 에러")
