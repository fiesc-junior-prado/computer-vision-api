from fastapi import APIRouter, File, UploadFile, HTTPException
from app.services.visualization import draw_yolo_boxes
from fastapi.responses import Response 

router = APIRouter(
    prefix="/visualization",
    tags=["visualização"]
)

@router.post("/", summary="Desenha bounding boxes YOLO em uma imagem")
async def upload_file(
    image: UploadFile = File(..., description="Arquivi de imagem(jpg, png, etc.)"),
    labels: UploadFile = File(..., description="Arquivo de rótulos YOLO (.txt)"),
):
    """
    Endpoint to upload an image file for visualization
    """
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    if not labels.content_type == "text/plain":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a YOLO labels file.")

    # carragando a imagem e o arquivo de rótulos
    try:
        img_bytes = await image.read()
        txt_bytes = await labels.read()
        # converter bytes para txt
        txt_content = txt_bytes.decode("utf-8")
        
        # CHAMA FUNÇÃO PARA DESENHAR AS BOUNDING BOXES
        result_image_bytes = draw_yolo_boxes(img_bytes, txt_content)
        # ------- para download da imagem com bounding boxes desenhadas
        # return Response(content=result_image_bytes, media_type="image/jpeg",headers={"Content-Disposition": f"attachment; filename=visualized_{image.filename}"})
        # ------- para visualização da imagem com bounding boxes desenhadas no navegador
        return Response(content=result_image_bytes, media_type="image/jpeg",headers={"Content-Disposition": f"inline; filename=visualized_{image.filename}"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno ao processar os arquivos: {str(e)}")