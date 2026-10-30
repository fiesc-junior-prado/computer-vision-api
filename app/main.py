import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import conversion, visualization, classification, detection

app = FastAPI(
    title="Computer Vision API",
    description="API para tarefas e utilitários de Visão Computacional",
    version="1.0.0",
    # lifespan=lifespan
)
# main.py no seu FastAPI
origins = [
    "http://localhost:5173", # Mantém para testes locais
    "https://visionc-tau.vercel.app/" # <--- O novo link do frontend em produção
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["Health"])
async def health():
    return {"status": "ok"}

#adicionando rotas
app.include_router(conversion.router)
app.include_router(visualization.router)
app.include_router(classification.router)
app.include_router(detection.router)

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",       
        host="0.0.0.0",       
        port=10000,            
        reload=True,          
    )