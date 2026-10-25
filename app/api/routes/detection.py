from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import Response
from app.services.detection import predict_to_yolo_format
import os

router = APIRouter(prefix="/detection", tags=["Object Detection"])

@router.post("/detect", summary="Detecta objetos em uma imagem e retorna anotações no formato YOLO")
def detect_objects(
    file: UploadFile = File(...),
    version: str = "v8"  # Deve ser 'v8', 'v9' ou 'v11'
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato inválido. Envie uma imagem.")
        
    print(f"Recebido arquivo: {file.filename}, tipo: {file.content_type}, versão YOLO (ONNX): {version}")
    
    try:
        # 1. Leitura síncrona dos bytes usando o handler subjacente do UploadFile
        image_bytes = file.file.read()
        
        # 2. Gerar o conteúdo do YOLO TXT (Carregará o modelo ONNX aqui se for a 1ª vez)
        txt_content = predict_to_yolo_format(image_bytes, version)
        
        # 3. Extrair o nome original da imagem e trocar a extensão para .txt
        nome_original = file.filename
        nome_base = os.path.splitext(nome_original)[0]
        nome_txt = f"{nome_base}.txt"
        
        # 4. Retornar a resposta
        return Response(
            content=txt_content,
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{nome_txt}"'}
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except FileNotFoundError as fnfe:
        raise HTTPException(status_code=404, detail=str(fnfe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento: {str(e)}")