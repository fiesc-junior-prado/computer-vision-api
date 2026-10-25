from app.models.schemas import CocoDataset
from collections import defaultdict
import io
import zipfile

def convert_coco_to_yolo(coco_data: CocoDataset):

    # mapeamento das imagens pelo id para fácil acesso
    images_map = {image.id: image for image in coco_data.images}

    # mapeamento das anotações por image_id para fácil acesso
    annotations_by_image = defaultdict(list)
    for annotation in coco_data.annotations:
        annotations_by_image[annotation.image_id].append(annotation)

    zip_buffer = io.BytesIO()
    # Criação do arquivo zip em memória
    with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
        for image_id, image_info in images_map.items():
            img_width = image_info.width
            img_height = image_info.height
            yolo_lines=[]
            # criação do arquivo de texto para cada imagem
            for ann in annotations_by_image.get(image_id, []):
                # normalização das coordenadas da bounding box para o formato YOLO
                 # [ponto_x_minimo (canto superior esquerdo), ponto_y_minimo, largura_da_caixa, altura_da_caixa]
                x_min, y_min, bbox_w, bbox_h = ann.bbox
                class_id = ann.category_id
                x_center = (x_min + bbox_w / 2.0) / img_width
                y_center = (y_min + bbox_h / 2.0) / img_height
                
                # Normaliza também a largura e a altura totais da caixa
                norm_w = bbox_w / img_width
                norm_h = bbox_h / img_height
                
                # prevenindo assim erros fatais durante o treino do modelo YOLO.
                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))
                
                # Formata a linha com 6 casas decimais de precisão e adiciona à lista da imagem
                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
            txt_filename = f"{image_info.file_name.rsplit('.', 1)[0]}.txt"
            txt_content = "\n".join(yolo_lines)
            zip_file.writestr(txt_filename, txt_content)  
    #resetar o ponteiro do buffer para o início
    zip_buffer.seek(0)
    return zip_buffer