from fastapi import APIRouter
from app.models.schemas import CocoDataset
from fastapi.responses import StreamingResponse
from app.services.coco_to_yolo import convert_coco_to_yolo
router = APIRouter(
    prefix="/conversion",
    tags=["conversion"]
)

@router.post("/coco-to-yolo", summary="Convert COCO to YOLO format")
def coco_to_yolo(payload: CocoDataset):
    """
    Endpoint to convert COCO format annotations to YOLO format
    """
    zip_buffer = convert_coco_to_yolo(payload)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=converted_yolo.zip"}
    )