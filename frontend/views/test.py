import numpy as np
import json
import sys
import keras
sys.path.append('../../')
from src import embeddings
import time

def create():
    keras.backend.clear_session()    # 0.009s
    model = keras.models.load_model('facenet_keras.h5')    # 37.322s

    start2=time.time()
    emb = embeddings.get_embedding_from_one_pic(model, '손흥민.jpeg')    # 8.247s
    end2=time.time()

    faculty_json = open('faculty_emb.json',encoding='utf-8')    # 0.000s
    faculty_dict = json.load(faculty_json)    # 0.005s
    min_dist = 100
    name = ""

    # 0.002s
    for key in faculty_dict:    
        dist = np.linalg.norm(faculty_dict[key]-emb)
        if min_dist>dist:
            min_dist=dist
            name = key
    
    print(f'예측 결과 : 교수님 성함-{name}, in_dist-{min_dist}')   

if __name__=="__main__":
    create()    # 45.59s