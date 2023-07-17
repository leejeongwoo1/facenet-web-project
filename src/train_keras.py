import os
import numpy as np
from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow import keras
from embeddings import load_dataset, get_embedding
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
import joblib
import pickle

def get_data(model, train_path, val_path):
    trainX, trainy = load_dataset(train_path)
    testX, testy = load_dataset(val_path)

    newTrainX = []
    for face_pixels in trainX:#face_pixels = 이차원 array
        embedding = get_embedding(model, face_pixels)
        newTrainX.append(embedding)
    newTrainX = np.asarray(newTrainX)
    
    newTestX = []
    for face_pixels in testX:
        embedding = get_embedding(model, face_pixels)
        newTestX.append(embedding)
    newTestX = np.asarray(newTestX)
    
    return newTrainX, trainy, newTestX, testy
    
def main():
    print("##### load data and preprocess data #####")
    model = keras.models.load_model(os.path.abspath('../model/facenet_keras.h5'))
    trainX, trainy, testX, testy = get_data(model, 
                                    os.path.abspath('../dataset/train'),
                                    os.path.abspath('../dataset/val'))
    in_encoder = preprocessing.Normalizer(norm='l2')#방향벡터의 개념, 벡터를 정규화시켜서 거리가 1로 바꿔줌
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    
    out_encoder = preprocessing.LabelEncoder()#label을 숫자로 변경시켜줌
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    testy = out_encoder.transform(testy)
    with open('../model/class_labeler.pkl','wb') as f:#open pkl and save out_encoder
        pickle.dump(out_encoder,f)

    # fit model
    print("##### fit model #####")
    classifier = svm.SVC(kernel='linear',probability=True)
    classifier.fit(trainX, trainy)
    
    # predict
    print("##### predict #####")
    yhat_train = classifier.predict(trainX)
    score_train = accuracy_score(trainy, yhat_train)
    
    yhat_test = classifier.predict(testX)
    print("label order: ")
    print(out_encoder.classes_)
    print("predict: ")
    print(yhat_test) 
    print("answer: ") 
    print(testy)
    score_test = accuracy_score(testy, yhat_test)
    print("Accuracy: train=%.3f, test=%.3f" % (score_train*100, score_test*100))
    joblib.dump(classifier, '../model/class_classifier.pkl')
    
    

if __name__ == '__main__':
    main()
