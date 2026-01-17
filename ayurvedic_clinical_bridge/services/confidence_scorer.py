"""
Confidence Scoring System for Knowledge Queries

This module provides dynamic confidence scoring based on:
1. Model performance metrics (F1-score, precision, recall)
2. Query-answer semantic similarity
3. Content quality assessment
4. Knowledge source reliability
"""

import json
import re
import math
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfidenceScorer:
    """Dynamic confidence scoring system for knowledge queries."""
    
    def __init__(self, model_metrics_path: Optional[str] = None):
        """Initialize confidence scorer with model performance data."""
        self.model_metrics_path = model_metrics_path or self._get_default_metrics_path()
        self.model_performance = self._load_model_metrics()
        
        # Base confidence levels for different content types
        self.base_confidence = {
            "curated_fundamental": 0.85,  # High confidence for curated fundamental concepts
            "curated_herb": 0.80,         # High confidence for curated herb information
            "curated_health": 0.75,       # Good confidence for curated health approaches
            "database_filtered": 0.65,    # Medium confidence for filtered database results
            "database_unfiltered": 0.45   # Lower confidence for unfiltered database results
        }
        
        logger.info("ConfidenceScorer initialized with model performance data")
    
    def _get_default_metrics_path(self) -> str:
        """Get default path to model metrics file."""
        base_dir = Path(__file__).parent.parent.parent
        return str(base_dir / "data" / "model_metrics.json")
    
    def _load_model_metrics(self) -> Dict[str, Any]:
        """Load model performance metrics from file."""
        try:
            with open(self.model_metrics_path, 'r') as f:
                data = json.load(f)
            
            # Extract performance metrics for the best performing model
            models = data.get('models', [])
            if not models:
                logger.warning("No model metrics found, using default values")
                return self._get_default_performance()
            
            # Find best model by F1-score
            best_model = max(models, key=lambda m: m.get('f1_score', 0))
            
            performance = {
                'f1_score': best_model.get('f1_score', 0.75),
                'precision': best_model.get('precision', 0.77),
                'recall': best_model.get('recall', 0.75),
                'accuracy': best_model.get('accuracy', 0.95),
                'model_name': best_model.get('model_name', 'BiLSTM-CRF'),
                'validation_loss': best_model.get('validation_loss', [0.1])[-1]  # Final validation loss
            }
            
            logger.info(f"Loaded performance metrics for {performance['model_name']}: F1={performance['f1_score']:.3f}")
            return performance
            
        except Exception as e:
            logger.error(f"Failed to load model metrics: {str(e)}")
            return self._get_default_performance()
    
    def _get_default_performance(self) -> Dict[str, Any]:
        """Get default performance metrics if file loading fails."""
        return {
            'f1_score': 0.756,
            'precision': 0.770,
            'recall': 0.755,
            'accuracy': 0.960,
            'model_name': 'BiLSTM-CRF',
            'validation_loss': 0.097
        }
    
    def calculate_confidence(
        self,
        query_text: str,
        concepts: List[Dict[str, Any]],
        content_type: str,
        processing_time: float,
        metadata: Dict[str, Any] = None
    ) -> float:
        """
        Calculate dynamic confidence score based on multiple factors.
        
        Args:
            query_text: Original query text
            concepts: List of returned concepts
            content_type: Type of content (curated_fundamental, curated_herb, etc.)
            processing_time: Time taken to process query
            metadata: Additional metadata about the query/response
        
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not concepts:
            return 0.0
        
        # Start with base confidence for content type
        base_conf = self.base_confidence.get(content_type, 0.5)
        
        # Factor 1: Model performance adjustment
        model_factor = self._calculate_model_performance_factor()
        
        # Factor 2: Query-answer semantic matching
        semantic_factor = self._calculate_semantic_matching_factor(query_text, concepts)
        
        # Factor 3: Content quality assessment
        quality_factor = self._calculate_content_quality_factor(concepts)
        
        # Factor 4: Processing efficiency factor
        efficiency_factor = self._calculate_efficiency_factor(processing_time, len(concepts))
        
        # Factor 5: Query specificity factor
        specificity_factor = self._calculate_query_specificity_factor(query_text)
        
        # Combine factors using weighted average
        weights = {
            'base': 0.3,
            'model': 0.2,
            'semantic': 0.2,
            'quality': 0.15,
            'efficiency': 0.1,
            'specificity': 0.05
        }
        
        final_confidence = (
            weights['base'] * base_conf +
            weights['model'] * model_factor +
            weights['semantic'] * semantic_factor +
            weights['quality'] * quality_factor +
            weights['efficiency'] * efficiency_factor +
            weights['specificity'] * specificity_factor
        )
        
        # Ensure confidence is within bounds
        final_confidence = max(0.1, min(0.98, final_confidence))
        
        logger.debug(f"Confidence calculation: base={base_conf:.3f}, model={model_factor:.3f}, "
                    f"semantic={semantic_factor:.3f}, quality={quality_factor:.3f}, "
                    f"efficiency={efficiency_factor:.3f}, specificity={specificity_factor:.3f}, "
                    f"final={final_confidence:.3f}")
        
        return round(final_confidence, 3)
    
    def _calculate_model_performance_factor(self) -> float:
        """Calculate confidence factor based on model performance."""
        # Use F1-score as primary metric, adjusted by precision and recall balance
        f1 = self.model_performance['f1_score']
        precision = self.model_performance['precision']
        recall = self.model_performance['recall']
        
        # Penalize if precision and recall are very imbalanced
        balance_penalty = abs(precision - recall) * 0.5
        
        # Convert F1-score to confidence factor (F1 typically 0.6-0.9 -> factor 0.7-1.0)
        model_factor = min(1.0, f1 + 0.1 - balance_penalty)
        
        return model_factor
    
    def _calculate_semantic_matching_factor(self, query: str, concepts: List[Dict[str, Any]]) -> float:
        """Calculate semantic matching between query and response."""
        if not concepts:
            return 0.0
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        best_match_score = 0.0
        
        for concept in concepts:
            concept_name = concept.get('concept_name', '').lower()
            descriptions = concept.get('descriptions', [])
            ayurvedic_terms = concept.get('ayurvedic_terms', [])
            
            # Combine all text for matching
            all_text = concept_name + ' ' + ' '.join(descriptions[:2])  # First 2 descriptions
            all_text += ' ' + ' '.join([term.lower() for term in ayurvedic_terms])
            
            concept_words = set(re.findall(r'\b\w+\b', all_text.lower()))
            
            # Calculate word overlap
            if query_words and concept_words:
                overlap = len(query_words.intersection(concept_words))
                total_unique = len(query_words.union(concept_words))
                jaccard_similarity = overlap / total_unique if total_unique > 0 else 0
                
                # Boost for exact phrase matches
                phrase_boost = 0.0
                for word in query_words:
                    if len(word) > 3 and word in all_text:
                        phrase_boost += 0.1
                
                match_score = min(1.0, jaccard_similarity + phrase_boost)
                best_match_score = max(best_match_score, match_score)
        
        # Convert to confidence factor (0.0-1.0 -> 0.5-1.0)
        return 0.5 + (best_match_score * 0.5)
    
    def _calculate_content_quality_factor(self, concepts: List[Dict[str, Any]]) -> float:
        """Calculate content quality factor based on completeness and structure."""
        if not concepts:
            return 0.0
        
        quality_scores = []
        
        for concept in concepts:
            score = 0.0
            
            # Factor 1: Concept name quality
            concept_name = concept.get('concept_name', '')
            if len(concept_name) > 10:
                score += 0.2
            if len(concept_name) > 30:
                score += 0.1
            
            # Factor 2: Description quality
            descriptions = concept.get('descriptions', [])
            if descriptions:
                avg_desc_length = sum(len(desc) for desc in descriptions) / len(descriptions)
                if avg_desc_length > 50:
                    score += 0.3
                if len(descriptions) >= 3:
                    score += 0.2
            
            # Factor 3: Ayurvedic terms presence
            ayurvedic_terms = concept.get('ayurvedic_terms', [])
            if ayurvedic_terms:
                score += 0.2
            
            # Factor 4: Sources presence
            sources = concept.get('sources', [])
            if sources:
                score += 0.1
            
            # Factor 5: No fragmentation indicators
            if not any(indicator in concept_name for indicator in ['A1:', 'Q:', '(', '?']):
                score += 0.1
            
            quality_scores.append(min(1.0, score))
        
        # Return average quality score
        return sum(quality_scores) / len(quality_scores) if quality_scores else 0.5
    
    def _calculate_efficiency_factor(self, processing_time: float, num_results: int) -> float:
        """Calculate efficiency factor based on processing time and results."""
        # Ideal processing time: < 0.1s for curated, < 1s for database search
        if processing_time < 0.001:  # Curated response
            time_factor = 1.0
        elif processing_time < 0.1:
            time_factor = 0.95
        elif processing_time < 1.0:
            time_factor = 0.85
        elif processing_time < 5.0:
            time_factor = 0.75
        else:
            time_factor = 0.65
        
        # Results factor: optimal is 1-5 results
        if 1 <= num_results <= 3:
            results_factor = 1.0
        elif 4 <= num_results <= 5:
            results_factor = 0.95
        elif num_results == 0:
            results_factor = 0.0
        else:
            results_factor = 0.8  # Too many results might indicate less precision
        
        return (time_factor + results_factor) / 2
    
    def _calculate_query_specificity_factor(self, query: str) -> float:
        """Calculate factor based on query specificity."""
        query_lower = query.lower()
        
        # Specific terms boost confidence
        specific_terms = [
            'dosha', 'vata', 'pitta', 'kapha', 'panchakarma', 'agni', 'ojas',
            'turmeric', 'ashwagandha', 'brahmi', 'neem', 'ginger',
            'ayurveda', 'rasayana', 'pranayama'
        ]
        
        specificity_score = 0.5  # Base score
        
        for term in specific_terms:
            if term in query_lower:
                specificity_score += 0.1
        
        # Longer, more detailed queries get higher scores
        word_count = len(query.split())
        if word_count >= 4:
            specificity_score += 0.1
        if word_count >= 6:
            specificity_score += 0.1
        
        return min(1.0, specificity_score)
    
    def get_confidence_explanation(self, confidence: float) -> str:
        """Get human-readable explanation of confidence level."""
        if confidence >= 0.9:
            return "Very High - Curated content from authoritative sources"
        elif confidence >= 0.8:
            return "High - Well-matched content with good model performance"
        elif confidence >= 0.7:
            return "Good - Relevant content with moderate confidence"
        elif confidence >= 0.6:
            return "Moderate - Content found but with some uncertainty"
        elif confidence >= 0.5:
            return "Low - Limited matching or lower quality content"
        else:
            return "Very Low - Minimal confidence in results"
    
    def update_model_performance(self, new_metrics: Dict[str, Any]) -> None:
        """Update model performance metrics (for online learning scenarios)."""
        self.model_performance.update(new_metrics)
        logger.info(f"Updated model performance metrics: F1={self.model_performance.get('f1_score', 0):.3f}")

# Global instance
_global_confidence_scorer = None

def get_confidence_scorer() -> ConfidenceScorer:
    """Get or create global confidence scorer instance."""
    global _global_confidence_scorer
    if _global_confidence_scorer is None:
        _global_confidence_scorer = ConfidenceScorer()
    return _global_confidence_scorer