import os 
import cv2 as cv
import numpy as np
import tensorflow as tf

#Caminho dinâmico
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR,"api","weights","classification","modelo_cifar10.keras")

# Classes extraídas do seu treinamento CIFAR-10
CLASSES = {
    0: 'Avião', 1: 'Carro', 2: 'Pássaro', 3: 'Gato', 4: 'Cervo',
    5: 'Cachorro', 6: 'Sapo', 7: 'Cavalo', 8: 'Navio', 9: 'Caminhão'
}

model_tf_cifar10= tf.keras.models.load_model(MODEL_PATH)

def prever_imagem(image_bytes: bytes):
    # decodificar
    np_arr= np.frombuffer(image_bytes, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)

    if img is None:
        raise ValueError("Imagem inv´lida ou corrompida")

    #converter cores
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    #redimensionar
    img_resized = cv.resize(img_rgb,(32,32))

    # normalização
    img_namalized = img_resized/255.0

    # add dimensão
    img_array= np.expand_dims(img_namalized, axis=0)

    #Previsão
    previsoes= model_tf_cifar10.predict(img_array)

    # Obeter indice melho chance
    indice_class = int(np.argmax(previsoes[0]))
    confianca = float(previsoes[0][indice_class])

    return {
        "id_classe": indice_class,
        "resultado_class": CLASSES.get(indice_class,"Desconhecido"),
        "confianca": f"{round(confianca*100,2)}%",
    }