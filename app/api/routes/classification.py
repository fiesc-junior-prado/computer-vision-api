from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.classification import prever_imagem

router = APIRouter(
    prefix="/classification",
    tags=["classificação"]
)

@router.post("/cifar-10", summary="cassifica imagens", description=f"0: 'Avião', 1: 'Carro', 2: 'Pássaro', 3: 'Gato', 4: 'Cervo',    5: 'Cachorro', 6: 'Sapo', 7: 'Cavalo', 8: 'Navio', 9: 'Caminhão'")
async def classification_images(file: UploadFile = File(...)):
    # Valida imagem
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo não é uma umagem!")
    try:
        # Ler bytes do arquivo
        image_bytes = await file.read()
        #chamar serviço
        result = prever_imagem(image_bytes)
        
        return {
            "status": "sucesso",
            "predicao": result

        }
    
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Erro durante a inferencia: {str(e)}')
