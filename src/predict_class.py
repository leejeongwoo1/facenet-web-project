import numpy as np
from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow import keras
from embeddings import load_dataset, get_embedding, extract_face
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
import joblib
import pickle
import train_keras

def predict(filename):
  pixels = extract_face(filename)
  facenet_model = keras.models.load_model('C:/University/DAVI/facenet/fatigue_detector/model/facenet_keras.h5')
  fatigue_classifier_model = joblib.load('C:/University/DAVI/facenet/fatigue_detector/model/fatigue_classifier.pkl')
  fatigue_labeler_model = joblib.load('C:/University/DAVI/facenet/fatigue_detector/model/fatigue_labeler.pkl')
  newTrainX=[1]
  pos = get_embedding(facenet_model,pixels)
  newTrainX[0] = pos
  newTrainX = np.asarray(newTrainX)
  in_encoder = preprocessing.Normalizer(norm='l2')
  newTrainX = in_encoder.transform(newTrainX)
  return fatigue_labeler_model.classes_[fatigue_classifier_model.predict(newTrainX)]

if __name__=='__main__':
  print(predict('C:/University/DAVI/customdataset/predict/__7.jpg'))
  print(predict('C:/University/DAVI/customdataset/predict/1.jpg'))
  print(predict('C:/University/DAVI/customdataset/predict/1_.jpg'))
