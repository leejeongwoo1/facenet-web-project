# 1. cProfile : snakeviz profile.pstats, graphviz svg image
# 2. time.time()
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
from PIL import Image
import time

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def create():
    img = Image.open("../frontend/static/client_img/손흥민.jpeg")
    start1=time.time()
    img_cropped = mtcnn(img)
    end1=time.time()
    print(f"mtcnn으로 얼굴 찾기 : {end1-start1:.5f}")

    start2=time.time()
    emb = resnet(img_cropped.unsqueeze(0))[0].detach().numpy()
    end2=time.time()
    print(f"임베딩 벡터 만들기 : {end2-start2:.5f}")

    faculty_json = open('../model/faculty_emb.json',encoding='utf-8')
    faculty_dict = json.load(faculty_json)
    min_dist = 100
    name = ""

    start3=time.time()
    for key in faculty_dict:
        # print('faculty:',type(faculty_dict[key][0]))
        # print('emb:',type(emb[0]))
        dist = np.linalg.norm(faculty_dict[key]-emb)
        if min_dist>dist:
            min_dist=dist
            name = key
    end3=time.time()
    print(f"최소 찾기 : {end3-start3:.5f}")

    print(name, min_dist)
    department = name.split('_')[0]
    name_ = name.split('_')[1]
    faculty_path = "../frontend/static/faculty_img/"+name+".jpg"

if __name__=="__main__":
    start=time.time()
    create()
    end=time.time()
    print(f"전체 소요 시간 : {end-start:.5f}")