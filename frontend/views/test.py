import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
from PIL import Image
import time

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def create():
    start=time.time()

    img = Image.open("../static/client_img/손흥민.jpeg")
    img_cropped = mtcnn(img)

    emb = resnet(img_cropped.unsqueeze(0))[0].detach().numpy()
    faculty_json = open('../../model/faculty_emb.json',encoding='utf-8')
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
    faculty_path = "../static/faculty_img/"+name+".jpg"

    end=time.time()
    print(f"전체 소요 시간 : {end-start:.5f}")

if __name__=="__main__":
    create()