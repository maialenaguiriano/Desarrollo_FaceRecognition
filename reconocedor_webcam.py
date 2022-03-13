import base64
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from io import BytesIO
from skimage import io
import mtcnn
import cv2
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import pickle
import cv2
from skimage.io import imread, imshow

# Importamos el modelo seleccionado
filename = 'selected_model.sav'
gscv = pickle.load(open('../data/' + filename, 'rb'))

data = np.load('../data/data-faces-embeddings.npz')
y_train= data['arr_1']
out_encoder = LabelEncoder()
out_encoder.fit(y_train)

# Volvemos a utilizar la función de la representación vectorial de las imágenes
def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    sample = np.expand_dims(face, axis=0)
    yhat = model.predict(sample)
    return yhat[0]

# Cargamos el modelo de facenet otra vez
facenet_model = load_model('../input/model/facenet_keras.h5')

in_encoder = Normalizer()

# Función que te realiza la predicción automáticamente después de hacerte la foto
def webcam_img_predict():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Pulsa SPACE para hacer una foto y ESC para cerrar la ventana")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error en la cámara")
            break
        cv2.imshow("Pulsa SPACE para hacer una foto y ESC para cerrar la ventana", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # Cuando presionas el botón ESC, se cierra la ventana
            print("Cerrando...")
            break
        elif k%256 == 32:
            # Cuando presionas el botón SPACE, se hace la foto
            print("Imagen cargada correctamente!")

    cam.release()

    cv2.destroyAllWindows()

    try:
        pixels = frame
        # create the detector, using default weights
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # deal with negative pixel index
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize((160, 160))
        face_array = np.asarray(image)
        emb_fotoprueba = get_embedding(facenet_model, face_array)
        emb_fotoprueba = emb_fotoprueba.reshape(1, -1)
        emb_fotoprueba_norm = in_encoder.transform(emb_fotoprueba)
        yhat_class = gscv.predict(emb_fotoprueba_norm)
        yhat_prob = gscv.predict_proba(emb_fotoprueba_norm)
        # get name
        class_index = yhat_class[0]
        class_probability = yhat_prob[0,class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)

        print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
        # print('Predicted probabilities: \n%s \n%s' % (all_names, yhat_prob[0]*100))
        plt.imshow(image)
        title = '%s (%.3f)' % (predict_names[0], class_probability)
        plt.title(title)
        output = plt.show()
    except:
        output = 'La imagen que te has sacado no es correcta'
    
    return output

webcam_img_predict()