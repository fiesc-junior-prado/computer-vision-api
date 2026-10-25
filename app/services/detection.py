import os
import cv2
import numpy as np
from ultralytics import YOLO

# Dicionário global que será preenchido durante o lifespan
MODELS = {}

def load_models():
    """Carrega os pesos na inicialização da aplicação."""
    print("Iniciando carregamento dos modelos YOLO...")
    
    # Caminho absoluto baseado na estrutura do projeto (app/api/weights)
    base_dir = os.path.dirname(os.path.dirname(__file__)) # aponta para a pasta app/
    weights_dir = os.path.join(base_dir, "api", "weights","detection")
    
    # Certifique-se de que os nomes dos arquivos .pt correspondam aos que você tem lá
    MODELS["v8"] = YOLO(os.path.join(weights_dir, "yolov8.pt"))
    MODELS["v9"] = YOLO(os.path.join(weights_dir, "yolov9.pt"))
    MODELS["v11"] = YOLO(os.path.join(weights_dir, "yolo11.pt"))
    print("Modelos carregados com sucesso.")

def clear_models():
    """Limpa a memória no encerramento."""
    MODELS.clear()

def get_model(version: str) -> YOLO:
    """Recupera o modelo já instanciado da memória."""
    if version not in MODELS:
        raise ValueError(f"Modelo {version} não está carregado.")
    return MODELS[version]

def predict_to_yolo_format(image_bytes: bytes, version: str) -> str:
    """Executa a inferência e retorna a string no formato YOLO."""
    # 1. Substituição do PIL pelo OpenCV/NumPy para evitar travamentos de conversão e canais RGBA
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    model = get_model(version)

    # Executar a inferência diretamente
    results = model(image, conf=0.5)
    
    yolo_lines = []
    
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
            
        # 2. Iteração segura diretamente sobre os objetos da classe Box da Ultralytics
        for box in result.boxes:
            # Extrair a classe e garantir conversão para int
            class_id = int(box.cls[0].item())
            
            # Extrair as coordenadas normalizadas extraindo do primeiro item do tensor
            x_center, y_center, w, h = box.xywhn[0].tolist()
            
            # Formatar a string
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    return "\n".join(yolo_lines)