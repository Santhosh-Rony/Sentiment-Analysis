"""
FastAPI application for sentiment analysis using pre-trained DistilBERT model.
Note: Uses PyTorch backend for transformers (industry standard for pre-trained models).
You can still use TensorFlow + Keras for custom model training in your own projects.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
from transformers import pipeline
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    framework="pt"  # Use PyTorch backend (pre-trained models are in PyTorch format)
)

# Pydantic models
class PredictionRequest(BaseModel):
    text: str
    return_probabilities: Optional[bool] = False

class BatchRequest(BaseModel):
    texts: List[str]
    return_probabilities: Optional[bool] = False

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None

# Create FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready sentiment analysis using pre-trained DistilBERT",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the HTML UI."""
    with open("api/static/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """Predict sentiment for a single text."""
    try:
        # Validate input
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Get prediction
        result = sentiment_analyzer(request.text)[0]
        
        # Process result
        response = {
            "text": request.text,
            "sentiment": "Positive" if result["label"] == "POSITIVE" else "Negative",
            "confidence": result["score"]
        }

        # Add probabilities if requested
        if request.return_probabilities:
            response["probabilities"] = {
                "positive": result["score"] if result["label"] == "POSITIVE" else 1 - result["score"],
                "negative": result["score"] if result["label"] == "NEGATIVE" else 1 - result["score"]
            }

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    """Predict sentiment for multiple texts."""
    try:
        # Validate input
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        if len(request.texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Maximum 100 texts per batch")

        # Get predictions
        results = sentiment_analyzer(request.texts)
        
        # Process results
        processed_results = []
        for text, result in zip(request.texts, results):
            prediction = {
                "text": text,
                "sentiment": "Positive" if result["label"] == "POSITIVE" else "Negative",
                "confidence": result["score"]
            }
            
            if request.return_probabilities:
                prediction["probabilities"] = {
                    "positive": result["score"] if result["label"] == "POSITIVE" else 1 - result["score"],
                    "negative": result["score"] if result["label"] == "NEGATIVE" else 1 - result["score"]
                }
            
            processed_results.append(prediction)

        return {"predictions": processed_results, "total_processed": len(processed_results)}

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
        "backend": "pytorch",
        "note": "PyTorch for transformers, TensorFlow available for custom models"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)