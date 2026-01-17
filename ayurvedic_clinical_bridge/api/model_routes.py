"""
Model Comparison API Routes for Ayurvedic Clinical Bridge

This module provides API endpoints for model performance comparison and metrics.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
import logging
from datetime import datetime
import json
import os
from pathlib import Path

from ..middleware.auth_middleware import get_current_user
from ..models.user_models import User

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])

# Response Models
class ModelMetrics(BaseModel):
    model_name: str
    model_type: Literal['rnn', 'lstm', 'gru', 'bilstm', 'transformer']
    latency_ms: float
    memory_usage_mb: float
    training_loss: List[float]
    validation_loss: List[float]
    epochs: List[int]
    throughput_samples_per_sec: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    parameters_count: int
    training_time_hours: float
    training_time_seconds: float
    inference_time_ms: float
    gpu_memory_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

class ComparisonResponse(BaseModel):
    models: List[ModelMetrics]
    timestamp: str
    evaluation_dataset: str
    metadata: Dict[str, Any] = {}

def load_model_metrics() -> ComparisonResponse:
    """
    Load model metrics from actual training results.
    """
    metrics_file = Path("data/model_metrics.json")
    
    if not metrics_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model metrics not found. Please run training first using /api/models/trigger-training"
        )
    
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
            return ComparisonResponse(**data)
    except Exception as e:
        logger.error(f"Failed to load metrics file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model metrics: {str(e)}"
        )

@router.get("/comparison", response_model=ComparisonResponse)
async def get_model_comparison(
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get comprehensive model performance comparison data.
    
    Returns metrics for BiLSTM, Transformer, and Hybrid models including:
    - Latency and throughput metrics
    - Memory usage statistics
    - Training loss curves over epochs
    - Accuracy, precision, recall, and F1 scores
    - Resource utilization data
    """
    try:
        comparison_data = load_model_metrics()
        logger.info(f"Model comparison data retrieved: {len(comparison_data.models)} models")
        return comparison_data
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is (like 404)
        raise
    except Exception as e:
        logger.error(f"Failed to get model comparison: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model comparison: {str(e)}"
        )

@router.get("/metrics/{model_name}")
async def get_model_metrics(
    model_name: str,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get detailed metrics for a specific model.
    """
    try:
        comparison_data = load_model_metrics()
        
        # Find the requested model
        model_metrics = None
        for model in comparison_data.models:
            if model.model_name.lower() == model_name.lower():
                model_metrics = model
                break
        
        if not model_metrics:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_name}"
            )
        
        return model_metrics
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metrics for model {model_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model metrics: {str(e)}"
        )



@router.get("/benchmark")
async def get_benchmark_results(
    metric: Optional[str] = None,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get benchmark results comparing all models on specific metrics.
    """
    try:
        comparison_data = load_model_metrics()
        
        if metric:
            # Return specific metric comparison
            metric_data = {}
            for model in comparison_data.models:
                if hasattr(model, metric):
                    metric_data[model.model_name] = getattr(model, metric)
            
            if not metric_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid metric: {metric}"
                )
            
            # Find best and worst performers
            best_model = max(metric_data.items(), key=lambda x: x[1])
            worst_model = min(metric_data.items(), key=lambda x: x[1])
            
            return {
                "metric": metric,
                "results": metric_data,
                "best_performer": {"model": best_model[0], "value": best_model[1]},
                "worst_performer": {"model": worst_model[0], "value": worst_model[1]},
                "timestamp": comparison_data.timestamp
            }
        else:
            # Return overall benchmark summary
            summary = {
                "total_models": len(comparison_data.models),
                "evaluation_dataset": comparison_data.evaluation_dataset,
                "timestamp": comparison_data.timestamp,
                "performance_summary": {}
            }
            
            # Calculate performance summaries
            metrics_to_compare = ['accuracy', 'f1_score', 'latency_ms', 'memory_usage_mb', 'throughput_samples_per_sec']
            
            for metric_name in metrics_to_compare:
                values = [getattr(model, metric_name) for model in comparison_data.models if hasattr(model, metric_name)]
                if values:
                    best_idx = values.index(max(values)) if metric_name not in ['latency_ms', 'memory_usage_mb'] else values.index(min(values))
                    summary["performance_summary"][metric_name] = {
                        "best_model": comparison_data.models[best_idx].model_name,
                        "best_value": values[best_idx],
                        "average": sum(values) / len(values),
                        "range": [min(values), max(values)]
                    }
            
            return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get benchmark results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get benchmark results: {str(e)}"
        )

@router.post("/trigger-training")
async def trigger_model_training(
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Trigger model training and generate new metrics.
    """
    try:
        logger.info("Training triggered via API")
        
        # Import real training function
        from ..training.training_with_metrics import run_real_training_with_metrics
        
        # Run training in background (in production, use Celery or similar)
        import threading
        
        def run_training():
            try:
                logger.info("Starting real background training...")
                run_real_training_with_metrics()
                logger.info("Real background training completed")
            except Exception as e:
                logger.error(f"Real background training failed: {e}")
        
        # Start training in background thread
        training_thread = threading.Thread(target=run_training)
        training_thread.daemon = True
        training_thread.start()
        
        return {
            "success": True, 
            "message": "Real model training started in background with actual datasets and models. Metrics will be updated upon completion.",
            "estimated_time_minutes": 15,
            "training_type": "real",
            "datasets": "AyurGenix CSV + Integrated Datasets",
            "models": ["Simple RNN", "LSTM", "GRU", "BiLSTM-CRF", "BioBERT-Transformer"]
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger training: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger training: {str(e)}"
        )

@router.get("/training-status")
async def get_training_status(
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Get current training status.
    """
    try:
        # Check if training is in progress (simplified check)
        import threading
        active_threads = [t for t in threading.enumerate() if 'training' in t.name.lower()]
        
        return {
            "training_in_progress": len(active_threads) > 0,
            "active_training_threads": len(active_threads),
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )



@router.post("/save-metrics")
async def save_model_metrics(
    metrics_data: ComparisonResponse,
    current_user: Optional[User] = Depends(lambda: None)
):
    """
    Save model metrics data (admin only).
    """
    try:
        # Check permissions
        if not current_user or not hasattr(current_user, 'role') or current_user.role != 'admin':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required to save model metrics"
            )
        
        # Ensure data directory exists
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        # Save metrics to file
        metrics_file = data_dir / "model_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data.dict(), f, indent=2)
        
        logger.info(f"Model metrics saved to {metrics_file}")
        return {"success": True, "message": f"Metrics saved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save model metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save model metrics: {str(e)}"
        )
