"""
Unified Model Manager for BioBERT and BiLSTM-CRF

This module provides a unified interface for managing and using both
BioBERT and BiLSTM-CRF models, with intelligent routing based on
user requirements and query complexity.
"""

import torch
from typing import Dict, List, Optional, Any, Union
import logging
import time
from enum import Enum
from dataclasses import dataclass

from .biobert_transformer import BioBERTClinicalProcessor, TaskType, ClinicalPrediction
from .bilstm_crf_processor import BiLSTMCRFClinicalProcessor

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types."""
    BIOBERT = "biobert"
    BILSTM_CRF = "bilstm_crf"
    AUTO = "auto"


class UserRole(Enum):
    """User role types for model selection."""
    GENERAL_USER = "general_user"
    QUALIFIED_PRACTITIONER = "qualified_practitioner"
    ADMIN = "admin"


@dataclass
class ModelSelectionCriteria:
    """Criteria for automatic model selection."""
    user_role: UserRole = UserRole.GENERAL_USER
    accuracy_preference: str = "balanced"  # "fast", "balanced", "high"
    query_complexity: str = "medium"  # "low", "medium", "high"
    include_safety_analysis: bool = True
    include_interactions: bool = True
    max_processing_time: Optional[float] = None


@dataclass
class ModelComparisonResult:
    """Result from model comparison."""
    biobert_result: Optional[ClinicalPrediction] = None
    bilstm_crf_result: Optional[ClinicalPrediction] = None
    recommended_model: ModelType = ModelType.AUTO
    comparison_metrics: Dict[str, Any] = None
    processing_times: Dict[str, float] = None


class UnifiedModelManager:
    """
    Unified manager for BioBERT and BiLSTM-CRF models.
    
    This manager provides intelligent routing between models based on
    user requirements, query complexity, and performance needs.
    """
    
    def __init__(self):
        """Initialize the unified model manager."""
        self.biobert_processor = None
        self.bilstm_crf_processor = None
        
        # Initialize processors
        try:
            self.biobert_processor = BioBERTClinicalProcessor()
            logger.info("BioBERT processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BioBERT processor: {e}")
        
        try:
            self.bilstm_crf_processor = BiLSTMCRFClinicalProcessor()
            logger.info("BiLSTM-CRF processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize BiLSTM-CRF processor: {e}")
        
        # Model selection rules
        self.selection_rules = self._initialize_selection_rules()
        
        logger.info(f"Unified model manager initialized - BioBERT: {'✅' if self.biobert_processor else '❌'}, BiLSTM-CRF: {'✅' if self.bilstm_crf_processor else '❌'}")
    
    def _initialize_selection_rules(self) -> Dict[str, Any]:
        """Initialize model selection rules."""
        return {
            'default_model': {
                UserRole.GENERAL_USER: ModelType.BILSTM_CRF,
                UserRole.QUALIFIED_PRACTITIONER: ModelType.BIOBERT,
                UserRole.ADMIN: ModelType.BIOBERT
            },
            'accuracy_preference': {
                'fast': ModelType.BILSTM_CRF,
                'balanced': ModelType.BILSTM_CRF,
                'high': ModelType.BIOBERT
            },
            'query_complexity': {
                'low': ModelType.BILSTM_CRF,
                'medium': ModelType.BILSTM_CRF,
                'high': ModelType.BIOBERT
            },
            'safety_critical': ModelType.BIOBERT,
            'interaction_analysis': ModelType.BIOBERT
        }
    
    def select_model(self, criteria: ModelSelectionCriteria) -> ModelType:
        """
        Select the most appropriate model based on criteria.
        
        Args:
            criteria: Selection criteria
            
        Returns:
            Selected model type
        """
        # Start with default model for user role
        selected_model = self.selection_rules['default_model'].get(
            criteria.user_role, ModelType.BILSTM_CRF
        )
        
        # Override based on accuracy preference
        if criteria.accuracy_preference in self.selection_rules['accuracy_preference']:
            accuracy_model = self.selection_rules['accuracy_preference'][criteria.accuracy_preference]
            if criteria.accuracy_preference == 'high':
                selected_model = accuracy_model
        
        # Override for high complexity queries
        if criteria.query_complexity == 'high':
            selected_model = ModelType.BIOBERT
        
        # Override for safety-critical analysis
        if criteria.include_safety_analysis and criteria.user_role == UserRole.QUALIFIED_PRACTITIONER:
            selected_model = ModelType.BIOBERT
        
        # Check model availability
        if selected_model == ModelType.BIOBERT and not self._is_biobert_available():
            logger.warning("BioBERT not available, falling back to BiLSTM-CRF")
            selected_model = ModelType.BILSTM_CRF
        
        if selected_model == ModelType.BILSTM_CRF and not self._is_bilstm_crf_available():
            logger.warning("BiLSTM-CRF not available, falling back to BioBERT")
            selected_model = ModelType.BIOBERT
        
        return selected_model
    
    def process_text(self, text: str, model_type: ModelType = ModelType.AUTO,
                    criteria: Optional[ModelSelectionCriteria] = None,
                    task_types: Optional[List[TaskType]] = None) -> ClinicalPrediction:
        """
        Process clinical text using the specified or automatically selected model.
        
        Args:
            text: Input clinical text
            model_type: Model to use (AUTO for automatic selection)
            criteria: Selection criteria for automatic model selection
            task_types: Tasks to perform
            
        Returns:
            Clinical prediction result
        """
        if criteria is None:
            criteria = ModelSelectionCriteria()
        
        if task_types is None:
            task_types = list(TaskType)
        
        # Select model if AUTO
        if model_type == ModelType.AUTO:
            model_type = self.select_model(criteria)
        
        # Process with selected model
        if model_type == ModelType.BIOBERT:
            return self._process_with_biobert(text, task_types)
        elif model_type == ModelType.BILSTM_CRF:
            return self._process_with_bilstm_crf(text, task_types)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def compare_models(self, text: str, task_types: Optional[List[TaskType]] = None) -> ModelComparisonResult:
        """
        Compare results from both models for the same input.
        
        Args:
            text: Input clinical text
            task_types: Tasks to perform
            
        Returns:
            Comparison result with both model outputs
        """
        if task_types is None:
            task_types = list(TaskType)
        
        result = ModelComparisonResult()
        result.processing_times = {}
        
        # Process with BioBERT
        if self._is_biobert_available():
            try:
                start_time = time.time()
                result.biobert_result = self._process_with_biobert(text, task_types)
                result.processing_times['biobert'] = time.time() - start_time
            except Exception as e:
                logger.error(f"BioBERT processing failed: {e}")
        
        # Process with BiLSTM-CRF
        if self._is_bilstm_crf_available():
            try:
                start_time = time.time()
                result.bilstm_crf_result = self._process_with_bilstm_crf(text, task_types)
                result.processing_times['bilstm_crf'] = time.time() - start_time
            except Exception as e:
                logger.error(f"BiLSTM-CRF processing failed: {e}")
        
        # Generate comparison metrics
        result.comparison_metrics = self._generate_comparison_metrics(result)
        
        # Recommend best model
        result.recommended_model = self._recommend_model_from_comparison(result)
        
        return result
    
    def _process_with_biobert(self, text: str, task_types: List[TaskType]) -> ClinicalPrediction:
        """Process text with BioBERT."""
        if not self._is_biobert_available():
            raise RuntimeError("BioBERT processor not available")
        
        return self.biobert_processor.process_clinical_text(text, task_types)
    
    def _process_with_bilstm_crf(self, text: str, task_types: List[TaskType]) -> ClinicalPrediction:
        """Process text with BiLSTM-CRF."""
        if not self._is_bilstm_crf_available():
            raise RuntimeError("BiLSTM-CRF processor not available")
        
        return self.bilstm_crf_processor.process_clinical_text(text, task_types)
    
    def _is_biobert_available(self) -> bool:
        """Check if BioBERT processor is available."""
        return self.biobert_processor is not None
    
    def _is_bilstm_crf_available(self) -> bool:
        """Check if BiLSTM-CRF processor is available."""
        return (self.bilstm_crf_processor is not None and 
                self.bilstm_crf_processor.is_available())
    
    def _generate_comparison_metrics(self, result: ModelComparisonResult) -> Dict[str, Any]:
        """Generate comparison metrics between models."""
        metrics = {}
        
        if result.biobert_result and result.bilstm_crf_result:
            # Entity count comparison
            biobert_entities = len(result.biobert_result.entities)
            bilstm_entities = len(result.bilstm_crf_result.entities)
            
            metrics['entity_counts'] = {
                'biobert': biobert_entities,
                'bilstm_crf': bilstm_entities,
                'difference': abs(biobert_entities - bilstm_entities)
            }
            
            # Processing time comparison
            if result.processing_times:
                metrics['processing_times'] = result.processing_times
                if 'biobert' in result.processing_times and 'bilstm_crf' in result.processing_times:
                    speedup = result.processing_times['biobert'] / result.processing_times['bilstm_crf']
                    metrics['speedup_factor'] = speedup
            
            # Confidence comparison
            biobert_avg_conf = sum(result.biobert_result.confidence_scores.values()) / len(result.biobert_result.confidence_scores)
            bilstm_avg_conf = sum(result.bilstm_crf_result.confidence_scores.values()) / len(result.bilstm_crf_result.confidence_scores)
            
            metrics['average_confidence'] = {
                'biobert': biobert_avg_conf,
                'bilstm_crf': bilstm_avg_conf
            }
        
        return metrics
    
    def _recommend_model_from_comparison(self, result: ModelComparisonResult) -> ModelType:
        """Recommend the best model based on comparison results."""
        if not result.biobert_result and not result.bilstm_crf_result:
            return ModelType.AUTO
        
        if not result.biobert_result:
            return ModelType.BILSTM_CRF
        
        if not result.bilstm_crf_result:
            return ModelType.BIOBERT
        
        # Both models available - make recommendation based on metrics
        metrics = result.comparison_metrics
        
        # If BiLSTM-CRF is significantly faster and entity counts are similar
        if (metrics.get('speedup_factor', 1) > 2 and 
            metrics.get('entity_counts', {}).get('difference', 0) <= 2):
            return ModelType.BILSTM_CRF
        
        # If BioBERT has significantly higher confidence
        biobert_conf = metrics.get('average_confidence', {}).get('biobert', 0)
        bilstm_conf = metrics.get('average_confidence', {}).get('bilstm_crf', 0)
        
        if biobert_conf > bilstm_conf + 0.1:
            return ModelType.BIOBERT
        
        # Default to BiLSTM-CRF for speed
        return ModelType.BILSTM_CRF
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all available models."""
        status = {
            'biobert': {
                'available': self._is_biobert_available(),
                'type': 'BioBERT Transformer',
                'use_case': 'High-accuracy clinical analysis',
                'target_users': ['qualified_practitioner', 'admin']
            },
            'bilstm_crf': {
                'available': self._is_bilstm_crf_available(),
                'type': 'BiLSTM-CRF',
                'use_case': 'Fast entity recognition',
                'target_users': ['general_user', 'qualified_practitioner']
            }
        }
        
        # Add detailed info if available
        if self._is_biobert_available():
            try:
                # Add BioBERT specific info if available
                status['biobert']['details'] = 'Loaded and ready'
            except Exception:
                pass
        
        if self._is_bilstm_crf_available():
            try:
                model_info = self.bilstm_crf_processor.get_model_info()
                status['bilstm_crf']['details'] = model_info
            except Exception:
                pass
        
        return status
    
    def get_recommendations(self, user_role: str, query_type: str = "general") -> Dict[str, Any]:
        """
        Get model recommendations for a specific user and query type.
        
        Args:
            user_role: User role (general_user, qualified_practitioner, admin)
            query_type: Type of query (general, clinical, safety, treatment)
            
        Returns:
            Recommendations with rationale
        """
        try:
            role_enum = UserRole(user_role)
        except ValueError:
            role_enum = UserRole.GENERAL_USER
        
        recommendations = {
            'primary_model': None,
            'fallback_model': None,
            'rationale': '',
            'expected_performance': {}
        }
        
        # Determine primary recommendation
        if role_enum == UserRole.QUALIFIED_PRACTITIONER:
            if query_type in ['clinical', 'safety', 'treatment']:
                recommendations['primary_model'] = 'biobert'
                recommendations['fallback_model'] = 'bilstm_crf'
                recommendations['rationale'] = 'BioBERT recommended for clinical analysis requiring high accuracy'
            else:
                recommendations['primary_model'] = 'bilstm_crf'
                recommendations['fallback_model'] = 'biobert'
                recommendations['rationale'] = 'BiLSTM-CRF recommended for fast general queries'
        else:
            recommendations['primary_model'] = 'bilstm_crf'
            recommendations['fallback_model'] = 'biobert'
            recommendations['rationale'] = 'BiLSTM-CRF recommended for general users (faster response)'
        
        # Add performance expectations
        recommendations['expected_performance'] = {
            'biobert': {
                'accuracy': 'High (90-95%)',
                'speed': 'Moderate (1-3s)',
                'best_for': 'Clinical analysis, safety assessment'
            },
            'bilstm_crf': {
                'accuracy': 'Good (85-90%)',
                'speed': 'Fast (0.2-0.8s)',
                'best_for': 'Entity recognition, general queries'
            }
        }
        
        return recommendations


# Global manager instance
_unified_manager = None

def get_unified_manager() -> UnifiedModelManager:
    """Get the global unified model manager instance."""
    global _unified_manager
    if _unified_manager is None:
        _unified_manager = UnifiedModelManager()
    return _unified_manager