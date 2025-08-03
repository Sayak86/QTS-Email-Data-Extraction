#!/usr/bin/env python3
"""
PII Analysis API Wrapper
FastAPI-based REST API for PII analysis service
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import yaml
import os
import asyncio
from datetime import datetime

from pii_pipeline import EnsemblePIIAnalyzer, PIIType, ConfidenceLevel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PII Analysis Service",
    description="Enterprise PII detection and masking service for UBS",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for UBS internal networks
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.ubs.com", "https://*.ubs.net"],  # Restrict to UBS domains
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global analyzer instance
analyzer = None

# Pydantic models for API
class PIIAnalysisRequest(BaseModel):
    """Request model for PII analysis"""
    text: str = Field(..., description="Text to analyze for PII", max_length=50000)
    config_override: Optional[Dict[str, Any]] = Field(None, description="Override default configuration")
    include_context: bool = Field(True, description="Include context around detected entities")
    mask_output: bool = Field(True, description="Return masked text")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Contact John Smith at john.smith@email.com or call 555-123-4567",
                "config_override": {"confidence_threshold": 0.7},
                "include_context": True,
                "mask_output": True
            }
        }

class PIIEntityResponse(BaseModel):
    """Response model for individual PII entity"""
    text: str
    type: str
    confidence: float
    confidence_level: str
    detector: str
    position: Dict[str, int]
    context: Optional[str] = None
    masked_value: str

class PIIAnalysisResponse(BaseModel):
    """Response model for PII analysis results"""
    request_id: str
    timestamp: str
    has_pii: bool
    total_entities: int
    high_confidence_entities: int
    processing_time: float
    entities: List[PIIEntityResponse]
    entities_by_type: Dict[str, int]
    masked_text: Optional[str] = None
    analyzer_stats: Dict[str, Any]
    
class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str
    version: str
    analyzers_available: List[str]
    system_info: Dict[str, Any]

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_path = os.getenv("PII_CONFIG_PATH", "config.yaml")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {
            "analyzers": {
                "presidio": {"enabled": True},
                "transformer": {"enabled": True},
                "spacy": {"enabled": True},
                "gliner": {"enabled": False}
            },
            "ensemble": {
                "voting_strategy": "majority",
                "confidence_threshold": 0.5
            }
        }

@app.on_event("startup")
async def startup_event():
    """Initialize the PII analyzer on startup"""
    global analyzer
    
    try:
        config = load_config()
        ensemble_config = config.get("ensemble", {})
        analyzer_config = config.get("analyzers", {})
        
        # Merge configurations
        full_config = {**ensemble_config, "analyzers": analyzer_config}
        
        analyzer = EnsemblePIIAnalyzer("api_ensemble", full_config)
        logger.info("PII Analyzer initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize PII analyzer: {e}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import psutil
    import sys
    
    analyzers_available = []
    if analyzer:
        analyzers_available = [a.name for a in analyzer.analyzers]
    
    return HealthResponse(
        status="healthy" if analyzer else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        analyzers_available=analyzers_available,
        system_info={
            "python_version": sys.version,
            "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "cpu_percent": psutil.cpu_percent()
        }
    )

@app.post("/analyze", response_model=PIIAnalysisResponse)
async def analyze_pii(request: PIIAnalysisRequest):
    """Analyze text for PII entities"""
    
    if not analyzer:
        raise HTTPException(status_code=503, detail="PII analyzer not available")
    
    try:
        # Generate request ID for tracking
        request_id = f"req_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(request.text) % 10000:04d}"
        
        # Apply configuration overrides if provided
        if request.config_override:
            # Create temporary analyzer with overridden config
            temp_config = analyzer.config.copy()
            temp_config.update(request.config_override)
            temp_analyzer = EnsemblePIIAnalyzer("temp", temp_config)
            result = temp_analyzer.analyze(request.text)
        else:
            result = analyzer.analyze(request.text)
        
        # Convert entities to response format
        entities_response = []
        for entity in result.entities:
            entity_response = PIIEntityResponse(
                text=entity.text,
                type=entity.pii_type.value,
                confidence=entity.confidence,
                confidence_level=entity.confidence_level.value,
                detector=entity.detector_name,
                position={"start": entity.start_pos, "end": entity.end_pos},
                context=entity.context if request.include_context else None,
                masked_value=entity.masked_value
            )
            entities_response.append(entity_response)
        
        # Count entities by type
        entities_by_type = {}
        for pii_type in PIIType:
            count = len(result.get_entities_by_type(pii_type))
            if count > 0:
                entities_by_type[pii_type.value] = count
        
        # Create response
        response = PIIAnalysisResponse(
            request_id=request_id,
            timestamp=datetime.utcnow().isoformat(),
            has_pii=result.has_pii,
            total_entities=len(result.entities),
            high_confidence_entities=len(result.high_confidence_entities),
            processing_time=result.processing_time,
            entities=entities_response,
            entities_by_type=entities_by_type,
            masked_text=result.masked_text if request.mask_output else None,
            analyzer_stats=result.detector_stats
        )
        
        # Log for audit trail
        logger.info(f"Request {request_id}: Analyzed {len(request.text)} characters, "
                   f"found {len(result.entities)} PII entities")
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/batch")
async def analyze_batch(texts: List[str], background_tasks: BackgroundTasks):
    """Analyze multiple texts in batch (async processing)"""
    
    if not analyzer:
        raise HTTPException(status_code=503, detail="PII analyzer not available")
    
    if len(texts) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    batch_id = f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%