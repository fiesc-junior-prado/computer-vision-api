import os
import cv2
import numpy as np
import gc
import onnxruntime as ort

# Caminhos dos modelos (mantendo a lógica do volume Docker)
BASE_LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LOCAL_WEIGHTS = os.path.join(BASE_LOCAL_DIR, "api", "weights")
WEIGHTS_DIR = os.getenv("MODELS_DIR", DEFAULT_LOCAL_WEIGHTS)

_models_cache = {}

MODEL_FILES = {
    "v8": "v8.onnx",
    "v9": "v9.onnx",
    "v11": "v11.onnx"
}

def get_model(version: str):
    """Carrega a sessão do ONNX Runtime com limites estritos de threads."""
    if version not in MODEL_FILES:
        raise ValueError(f"Versão '{version}' não é suportada.")
    
    if version not in _models_cache:
        model_path = os.path.join(WEIGHTS_DIR, MODEL_FILES[version])
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pesos não encontrados: {model_path}")
            
        print(f"Iniciando ONNX Runtime para {version} (CPU Limitada)...")
        
        # Configuração para evitar Deadlock no Docker Linux
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 1
        sess_options.inter_op_num_threads = 1
        
        _models_cache[version] = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"]
        )
        
    return _models_cache[version]

def clear_models():
    """Liberta a memória do ONNX Runtime."""
    _models_cache.clear()
    gc.collect()

def predict_to_yolo_format(image_bytes: bytes, version: str) -> str:
    """Inferência manual otimizada com OpenCV e ONNX."""
    session = get_model(version)
    
    # 1. Descodificação da imagem
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem inválida ou corrompida.")
        
    # 2. Pré-processamento (O YOLOv8/11 espera 1x3x640x640, RGB, normalizado 0-1)
    # A função blobFromImage do OpenCV faz o resize, swapRB (BGR para RGB) e normaliza (1/255.0) de forma altamente otimizada em C++
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True, crop=False)
    
    # 3. Execução da Inferência
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    output = outputs[0][0] # O formato de saída típico é [84, 8400]
    
    output = output.transpose() # Transpõe para [8400, 84] (8400 caixas, 4 coordenadas + 80 classes)
    
    boxes = []
    scores = []
    class_ids = []
    
    # 4. Filtragem inicial por confiança
    for row in output:
        classes_scores = row[4:]
        max_score = np.max(classes_scores)
        
        if max_score > 0.5:
            class_id = np.argmax(classes_scores)
            cx, cy, w, h = row[0:4]
            
            # O NMS do OpenCV exige [x_min, y_min, w, h] absolutos
            x_min = cx - (w / 2)
            y_min = cy - (h / 2)
            
            boxes.append([int(x_min), int(y_min), int(w), int(h)])
            scores.append(float(max_score))
            class_ids.append(class_id)
            
    # 5. Pós-processamento: Non-Maximum Suppression (Remove caixas sobrepostas do mesmo objeto)
    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.5, nms_threshold=0.4)
    
    # 6. Formatação final para texto YOLO
    txt_lines = []
    if len(indices) > 0:
        for i in indices.flatten():
            x_min, y_min, w, h = boxes[i]
            
            # O output do modelo ONNX é relativo à grelha de 640x640. 
            # Para o formato YOLO, temos de re-normalizar entre 0 e 1 dividindo por 640.
            cx_norm = (x_min + w / 2) / 640.0
            cy_norm = (y_min + h / 2) / 640.0
            w_norm = w / 640.0
            h_norm = h / 640.0
            
            txt_lines.append(f"{class_ids[i]} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
            
    return "\n".join(txt_lines)