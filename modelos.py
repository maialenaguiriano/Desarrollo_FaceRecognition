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

# Función para extraer cara de una imagen
def extraer_cara(idd):
    res = es.get(index = 'index_imagenes', id = idd)
    img_code = res['_source']['codificacion'].encode()
    img_label = res['_source']['nombre_persona']
    image = Image.open(BytesIO(base64.b64decode(img_code)))
    image = image.convert('RGB')
    image = ImageOps.exif_transpose(image) # evitamos que rote la imagen por cuestión de propiedades
    pixels = np.asarray(image)
    detector = MTCNN() # creamos el detector instalado previamente
    results = detector.detect_faces(pixels) # detecta las caras en la imagen
    # extrae las características de la cara
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2] # extrae la cara
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    return face_array, img_label

# Aplicamos la función a todas las imágenes para extraer cada una de las caras
res = es.search(index="index_imagenes")

faces = list()
labels = list()

for i in range(1, res["hits"]["total"]["value"] + 1):
    try:
        face = extraer_cara(i)[0]
        label = extraer_cara(i)[1]
        faces.append(face)
        labels.append(label)
    except:
        print(i)
        pass

X = np.asarray(faces)
y = np.asarray(labels)

# Dividimos las listas para train y test
seed = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = len(set(labels)), random_state = seed, stratify = y)
# guardamos y comprimimos los datos para posibles usos futuros
np.savez_compressed('../data/data.npz', X_train, y_train, X_test, y_test)

# Comprobamos la información guardada en train
fig = io.imshow_collection(X_train)
fig.set_figwidth(15)
fig.set_figheight(10)
print(y_train)
fig.savefig('../figures/X_train.jpeg')

# Hacemos lo mismo con test
fig = io.imshow_collection(X_test)
fig.set_figwidth(15)
fig.set_figheight(10)
print(y_test)
fig.savefig('../figures/X_test.jpeg')

# Comprobamos las dimensiones
print('Loaded: ', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Cargamos el modelo de facenet
facenet_model = load_model('../input/model/facenet_keras.h5')

# Creamos una representación vectorial de las imágenes mediante la función get_embedding() (una representación de menor dimensión)
def get_embedding(model, face):
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face-mean)/std
    sample = np.expand_dims(face, axis=0)
    yhat = model.predict(sample)
    return yhat[0]
    
# Aplicamos la función a cada imagen del train set
emdTrainX = list()
for face in X_train:
    emd = get_embedding(facenet_model, face)
    emdTrainX.append(emd)
    
emdTrainX = np.asarray(emdTrainX)
print(emdTrainX.shape)

# Aplicamos la función a cada imagen del test set
emdTestX = list()
for face in X_test:
    emd = get_embedding(facenet_model, face)
    emdTestX.append(emd)
emdTestX = np.asarray(emdTestX)
print(emdTestX.shape)

# guardamos y comprimimos los datos para posibles usos futuros
np.savez_compressed('../data/data-faces-embeddings.npz', emdTrainX, y_train, emdTestX, y_test)

# Dimensiones de los datos
print("Dataset: train=%d, test=%d" % (emdTrainX.shape[0], emdTestX.shape[0]))
# Normalizamos los vectores de entrada (X)
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)
# Normalizamos los labels (y)
out_encoder = LabelEncoder()
out_encoder.fit(y_train)
trainy_enc = out_encoder.transform(y_train)
testy_enc = out_encoder.transform(y_test)


####### PRIMER MODELO ---- LINEARSVC #######


# Construimos modelo
model = LinearSVC(random_state=seed, C= 2, class_weight='balanced')
model.fit(emdTrainX_norm, trainy_enc)

# Predecimos
pred_train = model.predict(emdTrainX_norm)
pred_test = model.predict(emdTestX_norm)

# Obtenemos scores (accuracy y f1)
accuracy_train = accuracy_score(trainy_enc, pred_train)
accuracy_test = accuracy_score(testy_enc, pred_test)

f1_train = f1_score(trainy_enc, pred_train, average = 'micro')
f1_test = f1_score(testy_enc, pred_test, average = 'micro')

# Resumen del modelo
print('Accuracy: train=%.3f, test=%.3f' % (accuracy_train*100, accuracy_test*100))
print('F1: train=%.3f, test=%.3f' % (f1_train*100, f1_test*100))

# Guardamos la informacion del modelo para indexarlo a Elasticsearch posteriormente
info_modelos = pd.DataFrame()
info_modelos.loc[0,'nombre'] = 'LinearSVC'
info_modelos.loc[0,'params'] = str(model.get_params())
info_modelos.loc[0,'accuracy_score'] = accuracy_test*100
info_modelos.loc[0,'f1_score'] = f1_test*100


####### SEGUNDO MODELO ---- SVC, linear kernel & GSCV #######

# Construimos modelo
model1 = SVC(kernel='linear', random_state=seed, probability = True)
param_grid = {'C': [0.001, 0.01, 0.1,0.2, 0.3, 0.5, 1.0, 5, 7, 13,40, 75,100], 'gamma': ('scale', 'auto', 1, 5, 10, 25, 50, 100)}
scorer = make_scorer(accuracy_score, greater_is_better=True)

# le digo el tipo de estimador, el tipo de parámetro, el tipo de cv y el tipo de scorer
# esto ya guarda todo, tendríamos 480 modelos
gscv = GridSearchCV(estimator=model1, param_grid=param_grid,cv=3,  scoring=scorer).fit(emdTrainX_norm, trainy_enc)

print('The best parameter combination is: ' + str(gscv.best_params_))
print('The corresponding CV score for them is: ' + str(gscv.best_score_))

# Predecimos
pred_train1 = gscv.best_estimator_.predict(emdTrainX_norm)
pred_test1 = gscv.best_estimator_.predict(emdTestX_norm)

# Obtenemos scores (accuracy y f1)
accuracy_train1 = accuracy_score(trainy_enc, pred_train1)
accuracy_test1 = accuracy_score(testy_enc, pred_test1)

f1_train1 = f1_score(trainy_enc, pred_train1, average = 'micro')
f1_test1 = f1_score(testy_enc, pred_test1, average = 'micro')

# Resumen del modelo
print('Accuracy: train=%.3f, test=%.3f' % (accuracy_train1*100, accuracy_test1*100))
print('F1: train=%.3f, test=%.3f' % (f1_train1*100, f1_test1*100))

# Guardamos la informacion del modelo
info_modelos.loc[1,'nombre'] = 'SVC_linear (best params)'
info_modelos.loc[1,'params'] = str(gscv.best_params_)
info_modelos.loc[1,'accuracy_score'] = accuracy_test1*100
info_modelos.loc[1,'f1_score'] = f1_test1*100

# Indexamos ambos modelos
es.indices.delete(index='index_modelos', ignore=[400, 404])

index = list(info_modelos.index)
idd = 1
for i in index:

    doc = {
        'nombre': info_modelos.loc[i,'nombre'],
        'parametros': info_modelos.loc[i,'params'],
        'accuracy_score': info_modelos.loc[i,'accuracy_score'],
        'f1_score': info_modelos.loc[i,'f1_score']
    }
    
    res = es.index(index = 'index_modelos', id = idd, document = doc)
    idd += 1

### Elegimos el segundo modelo y realizaremos las predicciones sobre él.
# Guardamos el modelo para importarlo a posteriori.
filename = 'selected_model.sav'
pickle.dump(gscv, open('../data/' + filename, 'wb'))