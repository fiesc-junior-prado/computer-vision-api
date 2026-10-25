import cv2 as cv
import numpy as np

def draw_yolo_boxes(image_bytes:bytes, txt_content:str):
    """
    Draw YOLO bounding boxes on the image.
    """
    # decodificar a imagem a partir dos bytes
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv.imdecode(np_arr, cv.IMREAD_COLOR)

    if img is None:
        raise ValueError("Erro ao decodificar a imagem. Certifique-se de que o arquivo de imagem é válido.")

    height, width, _ = img.shape

    # processar cada linha do arquivo de rótulos YOLO
    lines = txt_content.strip().split('\n')
    for line in lines:
        if not line.strip():
            continue  # pular linhas vazias
        parts = line.strip().split() 

        if len(parts) != 5:
            print(f"Formato de linha inválido: {line}. Cada linha deve conter 5 valores.")
            continue

        class_id = int(parts[0])
        x_center = float(parts[1]) 
        y_center = float(parts[2]) 
        w = float(parts[3]) 
        h = float(parts[4])

        #mapeando para pixels
        abs_x_center =int(x_center * width)
        abs_y_center =int(y_center * height)
        abs_w =int(w * width)
        abs_h =int(h * height)

        # calcular as coordenadas do retângulo
        x_min = int (abs_x_center - abs_w // 2)
        y_min = int (abs_y_center - abs_h // 2)
        x_max = int (abs_x_center + abs_w // 2)
        y_max = int (abs_y_center + abs_h // 2)

        # configurar a cor e a espessura do retângulo
        color = (0, 255, 0)  # verde
        thickness = 2
        
        #desenhar o retângulo na imagem
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

        # desenha o ID
        text = f"cls: {class_id}"
        cv.putText(img, text, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    # converter a imagem de volta para bytes
    success, img_encoded = cv.imencode('.jpg', img)
    if not success:
        raise ValueError("Erro ao codificar a imagem. Certifique-se de que a imagem é válida.")
    return img_encoded.tobytes()