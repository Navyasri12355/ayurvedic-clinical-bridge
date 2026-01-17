"""
FastAPI application with authentication and all route modules.

This module creates the complete REST API with authentication,
admin functionality, and all existing services.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

# Import route modules
from ayurvedic_clinical_bridge.api.auth_routes import router as auth_router
from ayurvedic_clinical_bridge.api.admin_routes import router as admin_router
from ayurvedic_clinical_bridge.api.knowledge_routes import router as knowledge_router
from ayurvedic_clinical_bridge.api.prescription_routes import router as prescription_router
from ayurvedic_clinical_bridge.api.safety_analysis_routes import router as safety_router
from ayurvedic_clinical_bridge.api.model_routes import router as model_router
from ayurvedic_clinical_bridge.api.medicine_mapping_routes import router as medicine_mapping_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Ayurvedic Clinical Bridge API - With Authentication",
    description="AI-driven clinical decision-support system with user authentication and role-based access",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(admin_router)
app.include_router(knowledge_router)
app.include_router(prescription_router)
app.include_router(safety_router)
app.include_router(model_router)
app.include_router(medicine_mapping_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Ayurvedic Clinical Bridge API - With Authentication",
        "version": "2.1.0",
        "status": "running",
        "features": [
            "User authentication with JWT tokens",
            "Role-based access control",
            "Practitioner credential verification",
            "Admin panel for user management",
            "Real knowledge system with compiled data",
            "Advanced prescription parsing with NER",
            "Comprehensive safety analysis",
            "Model performance metrics"
        ],
        "endpoints": {
            "auth": {
                "register": "/auth/register",
                "login": "/auth/login",
                "me": "/auth/me",
                "credentials": "/auth/credentials"
            },
            "admin": {
                "pending_practitioners": "/admin/pending-practitioners",
                "verify_practitioner": "/admin/verify-practitioner",
                "system_stats": "/admin/system-stats"
            },
            "api": {
                "knowledge": "/api/knowledge/query",
                "prescription": "/api/prescription/parse",
                "safety": "/api/safety-analysis/analyze",
                "models": "/api/models/metrics",
                "medicine_mapping": "/api/medicine-mapping/find-alternative"
            }
        }
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "authentication": "ready",
            "knowledge_system": "ready",
            "prescription_service": "ready", 
            "safety_analyzer": "ready",
            "admin_panel": "ready"
        }
    }

# Error handling
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail} - Path: {request.url.path}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "status_code": exc.status_code,
                "message": exc.detail,
                "path": str(request.url.path),
                "timestamp": time.time()
            }
        }
    )

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests."""
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,  # Use port 8000 for the authenticated version
        log_level="info",
        access_log=True,
        reload=True  # Enable reload for development
    )