import os
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from services.model_service import SignModelService, SIGNS
from services.sentence_service import SentenceService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SignConnectBackend")

app = FastAPI(
    title="SignConnect - Sign Language to Text Translator API",
    version="1.0.0",
    description="FastAPI backend providing real-time AI Sign Language recognition using Sign Transformer"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    model_service = SignModelService()
    logger.info(f"Model service initialized with backend: {model_service.backend_type}")
except Exception as e:
    logger.error(f"Error initializing model service: {e}")
    model_service = None

sentence_service = SentenceService()

# ----------------- Request Models -----------------
class PredictRequest(BaseModel):
    input: List[List[float]] = Field(
        ...,
        description="30 consecutive frames with 126 normalized landmark features per frame. Shape: (30, 126)"
    )

class SentenceRequest(BaseModel):
    words: List[str] = Field(
        ...,
        description="List of recognized sign tokens (e.g. ['HELLO', 'YOU'])"
    )
    mode: Optional[str] = Field(
        "nlp",
        description="Sentence generation mode: 'nlp', 'ai_model', or 'raw'"
    )
    language: Optional[str] = Field(
        "English",
        description="Target language for sentence translation/formatting"
    )
    api_key: Optional[str] = Field(
        None,
        description="Optional Groq API key for cloud AI model inference"
    )

# ----------------- API Endpoints -----------------
@app.get("/health")
def health_check():
    loaded = model_service.is_loaded() if model_service else False
    backend = model_service.backend_type if model_service else "none"
    return {
        "status": "online",
        "model_loaded": loaded,
        "backend": backend,
        "class_count": len(SIGNS),
        "primary_model": "sign_transformer.h5",
        "nlp_engine": "computational_g2t_nlp",
        "ai_models_supported": ["groq_llama_3.1", "local_nlp"]
    }

@app.get("/classes")
def get_classes():
    return {
        "classes": SIGNS,
        "count": len(SIGNS)
    }

@app.post("/predict")
def predict_sequence(req: PredictRequest):
    if not model_service or not model_service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recognition model is not loaded on the server."
        )

    try:
        result = model_service.predict(req.input)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@app.post("/sentence/build")
def build_sentence(req: SentenceRequest):
    try:
        result = sentence_service.build_sentence(
            words=req.words,
            mode=req.mode or "nlp",
            target_language=req.language or "English",
            api_key=req.api_key
        )
        return result
    except Exception as e:
        logger.error(f"Sentence building error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# ----------------- Static Frontend Hosting -----------------
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")

if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "SignConnect API is running. Frontend index.html not found."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
