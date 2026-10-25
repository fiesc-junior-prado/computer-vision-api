import os
import cv2
import numpy as np
import onnxruntime as ort

# Caminho dinâmico para o modelo
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Atualizado para buscar o arquivo .onnx
MODEL_PATH = os.path.join(BASE_DIR, "api", "weights", "modelo_classificacao.onnx")

# Carrega o modelo ONNX na memória iniciando a sessão
session = ort.InferenceSession(MODEL_PATH)

# Classes extraídas do treinamento CIFAR-10
CLASSES = {
    0: 'Avião', 1: 'Carro', 2: 'Pássaro', 3: 'Gato', 4: 'Cervo',
    5: 'Cachorro', 6: 'Sapo', 7: 'Cavalo', 8: 'Navio', 9: 'Caminhão'
}

def prever_imagem(image_bytes: bytes) -> dict:
    # 1. Decodificar a imagem recebida
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Imagem inválida ou corrompida.")
        
    # 2. Converter BGR (padrão OpenCV) para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Redimensionar para o formato do CIFAR-10 (32x32)
    img_resized = cv2.resize(img_rgb, (32, 32))
    
    # 4. Normalização idêntica ao treinamento (pixels de 0 a 1)
    img_normalized = img_resized / 255.0
    
    # 5. Adicionar a dimensão do batch e converter para float32 (obrigatório no ONNX)
    img_array = np.expand_dims(img_normalized, axis=0).astype(np.float32)
    
    # 6. Realizar a previsão utilizando o ONNX Runtime
    input_name = session.get_inputs()[0].name
    previsoes = session.run(None, {input_name: img_array})[0]
    
    # Obter o índice da classe com maior probabilidade
    indice_classe = int(np.argmax(previsoes[0]))
    confianca = float(previsoes[0][indice_classe])
    
    return {
        "id_classe": indice_classe,
        "nome_classe": CLASSES.get(indice_classe, "Desconhecido"),
        "confianca": round(confianca * 100, 2)
    }