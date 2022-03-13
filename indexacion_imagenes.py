import base64
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
from io import BytesIO
from skimage import io
from PIL import Image, ImageOps

# Instanciamos cliente Elasticsearch
es = Elasticsearch(hosts="http://192.168.56.108:9200/", timeout=30)

# Funcion para codificar imagen en BASE64
def codificada(imagen):
    with open(imagen, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    encoded_string = encoded_string.decode("utf-8")
    return encoded_string

# Creamos df de imágenes
imagenes = io.imread_collection("../images/*.jpeg")
files = []
personas = []
codificacion = []
estado_mascarilla = []
label = []
for image in imagenes.files:
    file = image.split("\\")[1]
    files.append(file)
    persona = image[image.find('images\\')+len('images\\'):image.rfind('_')]
    personas.append(persona)
    codificacion.append(codificada(image))
    if "bien" in image:
        estado_mascarilla.append("Bien puesta")
        label.append(0)
    elif "mal" in image:
        estado_mascarilla.append("Mal puesta")
        label.append(1)
    else:
        estado_mascarilla.append("Sin mascarilla")
        label.append(2)
    d = {'nombre_archivo': files, 'nombre_persona': personas, 'estado_mascarilla': estado_mascarilla, 'label': label, 'codificacion': codificacion}
    df_imagenes = pd.DataFrame(d)
    df_imagenes.to_csv('../data/csv_imagenes.csv', index = False)

# Mostramos las imagenes de la muestra
fig = io.imshow_collection(imagenes)
fig.set_figwidth(15)
fig.set_figheight(10)
fig.savefig('../figures/muestra.jpeg')

# Indexamos imágenes en un nuevo index (index_imagenes)
response = pd.read_csv('../data/csv_imagenes.csv')
index = list(response.index)
idd = 1
for i in index:

    doc = {
        'nombre_archivo': response.loc[i,'nombre_archivo'],
        'nombre_persona': response.loc[i,'nombre_persona'],
        'estado_mascarilla': response.loc[i,'estado_mascarilla'],
        'codificacion': response.loc[i,'codificacion']
    }
    
    res = es.index(index = 'index_imagenes', id = idd, document = doc)
    idd += 1

# Realizamos una query simple para comprobar que la indexación se ha hecho correctamente
res = es.search(index="index_imagenes", query={"match":{"estado_mascarilla":"Mal"}})
print(res['hits']['total']["value"])