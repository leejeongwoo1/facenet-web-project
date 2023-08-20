import os
from facenet_pytorch import MTCNN, InceptionResnetV1
import json
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def emb2json(face_path):

    dict = {}
    for i in os.listdir(face_path):

        full_path = os.path.join(face_path, i)
        face_info = i.split('.')[0]
        img = Image.open(full_path)
        img_cropped = mtcnn(img)
        dict[face_info] = resnet(img_cropped.unsqueeze(0))[0].tolist()

    for k, v in dict.items():
        ls = []
        for _v in v:
            ls.append(float(_v))
            dict[k] = ls

    with open('../model/face_dataset_emb.json', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)
    return dict


if __name__ == '__main__':
    emb2json('../frontend/static/face_dataset')