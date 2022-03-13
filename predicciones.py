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

# Instanciamos cliente Elasticsearch
es = Elasticsearch(hosts="http://192.168.56.108:9200/", timeout=30)

# Importamos el modelo seleccionado
filename = 'selected_model.sav'
gscv = pickle.load(open('../data/' + filename, 'rb'))

# Guardamos los resultados obtenidos en las predicciones realizadas con imágenes hechas en el momento desde la webcam. 
# Estos resultados variarían al volver a ejecutarse este script porque no podemos volver a repetir las fotos, 
# por lo que únicamente guardamos los resultados obtenidos para poder indexarlos en Elasticsearch

# Guardamos las predicciones con imágenes desde la webcam a mano
d = {'persona': ['Unai', 'Unai', 'Maialen', 'Maialen'], 'mascarilla': ['Sí', 'No', 'Sí', 'No'], 'prediccion': [1, 1, 1, 0], 'prob':[18.53, 13.07, 25.71, 9.67]}
df_predicciones = pd.DataFrame(d)
df_predicciones

# Cargamos la funcion de extraer_cara() modificada. Esta vez no obtenemos las imágenes desde Elasticsearch.
def extraer_cara(filename):
    image = Image.open(filename)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    return face_array


faces_prediction = list()
for filename in io.imread_collection("../predict_images/*.jpeg").files:
    face = extraer_cara(filename)
    faces_prediction.append(face)
faces_prediction = np.asarray(faces_prediction)

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

# Aplicamos la función de vectorización
to_predict = list()
for face in faces_prediction:
    emd = get_embedding(facenet_model, face)
    to_predict.append(emd)
to_predict = np.asarray(to_predict)


data = np.load('../data/data-faces-embeddings.npz') # si da error probar con el otro archivo comprimido
y_train= data['arr_1']
out_encoder = LabelEncoder()
out_encoder.fit(y_train)

# Realizamos las predicciones
for img in faces_prediction:
    emd = get_embedding(facenet_model, img)
    emd = emd.reshape(1, -1)
    in_encoder = Normalizer()
    img_norm = in_encoder.transform(emd)
    yhat_class = gscv.predict(img_norm)
    yhat_prob = gscv.predict_proba(img_norm)
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    plt.imshow(img)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    plt.title(title)
    plt.show()

# Guardamos las predicciones a mano (estos resultados podrían variar cada vez que se reejecute el script)
d2 = {'persona': ['Alba', 'Paule', 'Duran', 'Asier', 'Unax', 'Escalante', 'Garcia', 'Juan', 'Juan', 'June', 'Escalante', 'Alba', 'Duran', 'Garcia', 'Gabi', 'Gabi'], 'mascarilla': ['No', 'No', 'No', 'No', 'No', 'No', 'No', 'No', 'Sí', 'No', 'Sí', 'Sí', 'Sí', 'Sí', 'Sí', 'No'], 'prediccion': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], 'prob': [39.08, 40.04,24.00,17.25, 22.53,24.78,30.14,16.36,13.14,35.81,21.2,37.2,27.74,22.6,20.33,30.29]}
df_predicciones = df_predicciones.append(pd.DataFrame(d2), ignore_index = True)

# Indexamos las predicciones
index = list(df_predicciones.index)
idd = 1
for i in index:

    doc = {
        'persona': df_predicciones.loc[i,'persona'],
        'mascarilla': df_predicciones.loc[i,'mascarilla'],
        'prediccion': df_predicciones.loc[i,'prediccion'],
        'prob': df_predicciones.loc[i,'prob']
    }
    
    res = es.index(index = 'index_predicciones', id = idd, document = doc)
    idd += 1