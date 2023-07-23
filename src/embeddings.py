import PIL
import os
import numpy as np
from mtcnn import mtcnn
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import keras
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
    img = PIL.Image.open(face_path).convert('RGB')
    pixels = np.asarray(img)
    face_pixels = pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()

    face_pixels = (face_pixels - mean)/std
    samples = np.expand_dims(face_pixels, axis=0)

    yhat = model.predict(samples)
    return yhat[0]

if __name__ == '__main__':
    model = keras.models.load_model(os.path.abspath('../model/facenet_keras.h5'))
    embed = get_embedding_from_one_pic(model,'../dataset/faculty/face_img/소프트웨어융합학과_강형엽.jpg')
    print(embed)
