import PIL
import os
import numpy as np
from mtcnn import mtcnn
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import keras
import json

def extract_face(filename, size=(160,160)):
    img = PIL.Image.open(filename).convert('RGB')
    pixels = np.asarray(img)
    det = mtcnn.MTCNN()
    results = det.detect_faces(pixels)

    x1, y1 , width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    face = pixels[y1: y1+height, x1:x1+width]
    img = PIL.Image.fromarray(face)
    img = img.resize(size)

    return np.asarray(img)

def load_faces(directory):
    print("load face start")
    faces = []
    files = os.listdir(directory)#train/jin
    for filename in files:
        face = extract_face(os.path.join(directory, filename))
        faces.append(face)
    print("load face finished")
    return faces

def load_dataset(directory):
    print("load dataset start")
    X, y = list(), list()
    directory = os.path.abspath(directory)

    for subdir in os.listdir(directory):#train
        faces = os.path.join(directory, subdir)#train/jin
        faces = load_faces(faces)
        labels = [subdir for _ in range(len(faces))]
        X.extend(faces)
        y.extend(labels)
    print("load data finished")
    return np.asarray(X), np.asarray(y)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    
    face_pixels = (face_pixels - mean)/std
    samples = np.expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)
    return yhat[0]

def get_embedding_from_one_pic(model, face_path):
    img = PIL.Image.open(os.path.abspath(face_path)).convert('RGB')

    pixels = np.asarray(img)
    det = mtcnn.MTCNN()
    results = det.detect_faces(pixels)
    if len(results)==0:
        raise Exception("face is not detected")
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    face = pixels[y1: y1 + height, x1:x1 + width]
    img = PIL.Image.fromarray(face)
    img = img.resize((160,160))

    pixels = np.asarray(img)
    face_pixels = pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()

    face_pixels = (face_pixels - mean)/std
    samples = np.expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)
    return yhat[0]

def emb2json(face_path):
    model = keras.models.load_model('C:/University/23tgthon/model/facenet_keras.h5')
    dict = {}
    for i in os.listdir(face_path):
        full_path = os.path.join(face_path, i)
        face_info = i.split('.')[0]
        dict[face_info] = get_embedding_from_one_pic(model, full_path)

    for k, v in dict.items():
        ls = []
        for _v in v:
            ls.append(float(_v))
            dict[k] = ls

    with open('../model/faculty_emb.json', 'w', encoding='utf-8') as f:
        json.dump(dict, f, ensure_ascii=False, indent=4)
    return dict


if __name__ == '__main__':
    emb2json('../../23tgthon/dataset/faculty/origin_img')