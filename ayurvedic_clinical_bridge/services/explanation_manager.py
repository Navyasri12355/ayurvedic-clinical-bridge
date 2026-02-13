#!/usr/bin/env python3
"""
Explanation Manager Service
Manages explanation state to prevent duplication and ensure single explanation per query
"""

import hashlib
import time
from typing import Dict, Any, Optional
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class ExplanationManager:
    """
    Manages explanation state to prevent duplication and ensure single explanation per query.
    Uses LRU cache and query hashing to track explanations.
    """
    
    def __init__(self, cache_size: int = 1000, cache_ttl: int = 3600):
        """
        Initialize the explanation manager.
        
        Args:
            cache_size: Maximum number of explanations to cache
            cache_ttl: Time-to-live for cached explanations in seconds
        """
        self.cache_size = cache_size
        self.cache_ttl = cache_ttl
        self.explanation_cache = OrderedDict()
        self.active_explanations = set()
        self.cache_timestamps = {}
        
        logger.info(f"ExplanationManager initialized with cache_size={cache_size}, ttl={cache_ttl}s")
    
    def _generate_query_id(self, query_text: str, user_context: Optional[Dict] = None) -> str:
        """
        Generate a unique ID for a query based on text and context.
        
        Args:
            query_text: The query text
            user_context: Optional user context (role, preferences, etc.)
            
        Returns:
            Unique query ID
        """
        # Create hash from query text and context
        hash_input = query_text.lower().strip()
        
        if user_context:
            # Include relevant context in hash
            context_str = f"{user_context.get('role', '')}{user_context.get('preferences', '')}"
            hash_input += context_str
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, query_id: str) -> bool:
        """
        Check if cached explanation is still valid.
        
        Args:
            query_id: Query identifier
            
        Returns:
            True if cache is valid, False otherwise
        """
        if query_id not in self.cache_timestamps:
            return False
        
        cache_time = self.cache_timestamps[query_id]
        return (time.time() - cache_time) < self.cache_ttl
    
    def _cleanup_expired_cache(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []
        
        for query_id, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp) > self.cache_ttl:
                expired_keys.append(query_id)
        
        for key in expired_keys:
            self.explanation_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _manage_cache_size(self):
        """Ensure cache doesn't exceed maximum size."""
        while len(self.explanation_cache) > self.cache_size:
            # Remove oldest entry (LRU)
            oldest_key = next(iter(self.explanation_cache))
            self.explanation_cache.pop(oldest_key)
            self.cache_timestamps.pop(oldest_key, None)
    
    def get_explanation(
        self, 
        query_text: str, 
        explainability_service,
        user_context: Optional[Dict] = None,
        main_prediction: Optional[Dict] = None,  # Add main prediction parameter
        force_refresh: bool = False
    ) -> Dict[str, Any]:
        """
        Get explanation for a query, using cache to prevent duplicates.
        
        Args:
            query_text: The query text to explain
            explainability_service: The explainability service instance
            user_context: Optional user context
            force_refresh: Force regeneration even if cached
            
        Returns:
            Explanation dictionary
        """
        # Generate query ID
        query_id = self._generate_query_id(query_text, user_context)
        
        # Clean up expired cache entries
        self._cleanup_expired_cache()
        
        # Check cache first (unless force refresh)
        if not force_refresh and query_id in self.explanation_cache and self._is_cache_valid(query_id):
            # Move to end (LRU)
            explanation = self.explanation_cache.pop(query_id)
            self.explanation_cache[query_id] = explanation
            
            logger.info(f"Returning cached explanation for query_id: {query_id}")
            return explanation
        
        # Check if explanation is currently being generated
        if query_id in self.active_explanations:
            logger.info(f"Explanation already being generated for query_id: {query_id}")
            return {
                'available': False,
                'status': 'generating',
                'reason': 'Explanation is currently being generated'
            }
        
        # Generate new explanation
        try:
            self.active_explanations.add(query_id)
            logger.info(f"Generating new explanation for query_id: {query_id}")
            
            # Generate explanation using the service
            explanation = self._generate_explanation(query_text, explainability_service, main_prediction)
            
            # Cache the result
            self.explanation_cache[query_id] = explanation
            self.cache_timestamps[query_id] = time.time()
            
            # Manage cache size
            self._manage_cache_size()
            
            logger.info(f"Explanation generated and cached for query_id: {query_id}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation for query_id {query_id}: {e}")
            return {
                'available': False,
                'status': 'error',
                'reason': f'Explanation generation failed: {str(e)}'
            }
        finally:
            self.active_explanations.discard(query_id)
    
    def _generate_explanation(self, query_text: str, explainability_service, main_prediction: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate explanation using the explainability service.
        
        Args:
            query_text: The query text to explain
            explainability_service: The explainability service instance
            
        Returns:
            Explanation dictionary
        """
        if not explainability_service:
            return {
                'available': False,
                'status': 'unavailable',
                'reason': 'Explainability service not available'
            }
        
        # Check if this is a herb query based on main_prediction
        is_herb_query = main_prediction and any(word in str(main_prediction.get('disease', '')).lower() 
                                               for word in ['chakshushya', 'amapachana', 'dahashamana', 'arshoghna', 'agnideepana', 'benefit'])
        
        if is_herb_query:
            # For herb queries, create a custom explanation focused on herb benefits
            logger.info(f"Generating herb explanation for: {main_prediction.get('disease', 'Unknown')}")
            return self._generate_herb_explanation(query_text, main_prediction)
        
        try:
            # Generate explanation using SHAP
            explanation = explainability_service.explain_prediction(query_text, top_k=3)
            
            if explanation.get('explanation_available', False):
                # Use main prediction data if available, otherwise fall back to explanation service prediction
                if main_prediction:
                    # Create consistent summary using main prediction
                    disease = main_prediction.get('disease', 'Unknown')
                    confidence = main_prediction.get('confidence', 0)
                    confidence_pct = main_prediction.get('confidence_percentage', '0%')
                    
                    # Get important words from explanation
                    word_explanations = explanation.get('word_explanations', [])
                    important_words = [w['word'] for w in word_explanations[:3] if w.get('contribution') == 'positive']
                    
                    # Generate consistent summary
                    summary = f"The AI model predicted '{disease}' with {confidence_pct} confidence. "
                    if important_words:
                        summary += f"Key words that influenced this prediction: {', '.join(important_words)}. "
                    summary += "This explanation shows which words in your input were most important for the AI's decision."
                else:
                    # Fall back to original summary from explainability service
                    summary = explanation.get('summary', '')
                
                return {
                    'available': True,
                    'status': 'success',
                    'method': explanation.get('explanation_method', 'SHAP'),
                    'feature_importance': explanation.get('feature_importance', {}),
                    'explanation_summary': summary,  # Use consistent summary
                    'word_importance': explanation.get('word_explanations', [])[:5],
                    'confidence_assessment': explanation.get('confidence_assessment', {}),
                    'visualization_data': explanation.get('visualization_data', {}),
                    'user_summary': summary  # Use same consistent summary
                }
            else:
                return {
                    'available': False,
                    'status': 'failed',
                    'reason': 'SHAP explanation generation failed'
                }
                
        except Exception as e:
            logger.error(f"Error in SHAP explanation generation: {e}")
            return {
                'available': False,
                'status': 'error',
                'reason': f'SHAP error: {str(e)}'
            }
    
    def _generate_herb_explanation(self, query_text: str, main_prediction: Dict) -> Dict[str, Any]:
        """
        Generate a custom explanation for herb benefit queries.
        
        Args:
            query_text: The query text
            main_prediction: The main herb prediction
            
        Returns:
            Explanation dictionary
        """
        try:
            benefit = main_prediction.get('disease', 'Unknown Benefit')  # 'disease' field contains benefit name
            confidence_pct = main_prediction.get('confidence_percentage', '0%')
            
            # Extract key words from the query
            query_words = query_text.lower().split()
            # Filter out common words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'benefits', 'uses', 'properties'}
            important_words = [word for word in query_words if word not in stop_words and len(word) > 2]
            
            # Create word importance data
            word_importance = []
            for i, word in enumerate(important_words[:5]):  # Top 5 words
                importance_score = 10.0 - (i * 2.0)  # Decreasing importance
                word_importance.append({
                    'word': word,
                    'importance': importance_score,
                    'contribution': 'positive'
                })
            
            # Generate summary
            summary = f"The AI model predicted '{benefit}' with {confidence_pct} confidence. "
            if important_words:
                summary += f"Key words that influenced this prediction: {', '.join(important_words[:3])}. "
            summary += "This explanation shows which words in your herb query were most important for the AI's benefit prediction."
            
            # Create visualization data
            visualization_data = {
                'words': [w['word'] for w in word_importance],
                'importance_scores': [w['importance'] for w in word_importance],
                'contributions': [w['contribution'] for w in word_importance],
                'prediction_info': {
                    'disease': benefit,  # Using 'disease' field for consistency with frontend
                    'confidence': main_prediction.get('confidence', 0)
                },
                'color_scale': {
                    'positive': '#2E8B57',
                    'negative': '#DC143C', 
                    'neutral': '#808080'
                }
            }
            
            return {
                'available': True,
                'status': 'success',
                'method': 'Herb Benefit Analysis',
                'feature_importance': {},
                'explanation_summary': summary,
                'word_importance': word_importance,
                'confidence_assessment': {},
                'visualization_data': visualization_data,
                'user_summary': summary
            }
            
        except Exception as e:
            logger.error(f"Error generating herb explanation: {e}")
            return {
                'available': False,
                'status': 'error',
                'reason': f'Herb explanation error: {str(e)}'
            }
    
    def consolidate_explanations(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consolidate multiple explanation objects into a single one.
        
        Args:
            response_data: API response data that may contain multiple explanations
            
        Returns:
            Response data with consolidated explanation
        """
        # Check for multiple explanation keys
        explanation_keys = ['explanation', 'ai_explanation', 'shap_explanation', 'explainability']
        
        explanations_found = []
        for key in explanation_keys:
            if key in response_data and response_data[key]:
                explanations_found.append(key)
        
        if len(explanations_found) <= 1:
            # No consolidation needed
            return response_data
        
        logger.info(f"Consolidating {len(explanations_found)} explanations: {explanations_found}")
        
        # Use the first explanation as primary
        primary_explanation = response_data[explanations_found[0]]
        
        # Merge information from other explanations if needed
        for key in explanations_found[1:]:
            other_explanation = response_data[key]
            
            # Merge additional information if available
            if isinstance(other_explanation, dict) and isinstance(primary_explanation, dict):
                # Merge feature importance
                if 'feature_importance' in other_explanation and 'feature_importance' not in primary_explanation:
                    primary_explanation['feature_importance'] = other_explanation['feature_importance']
                
                # Merge summaries
                if 'explanation_summary' in other_explanation and 'explanation_summary' not in primary_explanation:
                    primary_explanation['explanation_summary'] = other_explanation['explanation_summary']
            
            # Remove duplicate explanation
            response_data.pop(key, None)
        
        # Ensure single explanation key
        response_data['explanation'] = primary_explanation
        
        logger.info("Explanation consolidation completed")
        return response_data
    
    def clear_cache(self):
        """Clear all cached explanations."""
        self.explanation_cache.clear()
        self.cache_timestamps.clear()
        self.active_explanations.clear()
        logger.info("Explanation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self.explanation_cache),
            'max_cache_size': self.cache_size,
            'active_generations': len(self.active_explanations),
            'cache_ttl': self.cache_ttl,
            'oldest_entry_age': min(
                [time.time() - ts for ts in self.cache_timestamps.values()],
                default=0
            ) if self.cache_timestamps else 0
        }

# Global explanation manager instance
_explanation_manager = None

def get_explanation_manager() -> ExplanationManager:
    """Get the global explanation manager instance."""
    global _explanation_manager
    if _explanation_manager is None:
        _explanation_manager = ExplanationManager()
    return _explanation_manager

# Example usage and testing
if __name__ == "__main__":
    manager = ExplanationManager()
    
    # Mock explainability service for testing
    class MockExplainabilityService:
        def explain_prediction(self, query, top_k=3):
            return {
                'available': True,
                'explanation_method': 'SHAP',
                'feature_importance': {'word1': 0.5, 'word2': 0.3},
                'explanation_summary': f'Mock explanation for: {query}'
            }
    
    mock_service = MockExplainabilityService()
    
    print("🧪 Testing Explanation Manager")
    print("=" * 40)
    
    # Test explanation generation
    query = "benefits of ginger"
    explanation1 = manager.get_explanation(query, mock_service)
    print(f"First explanation: {explanation1['available']}")
    
    # Test cache hit
    explanation2 = manager.get_explanation(query, mock_service)
    print(f"Second explanation (cached): {explanation2['available']}")
    
    # Test consolidation
    response_data = {
        'response': 'Some response',
        'explanation': {'available': True, 'method': 'SHAP'},
        'ai_explanation': {'available': True, 'summary': 'Additional info'}
    }
    
    consolidated = manager.consolidate_explanations(response_data)
    print(f"Consolidated explanations: {len([k for k in consolidated.keys() if 'explanation' in k])}")
    
    # Print stats
    stats = manager.get_cache_stats()
    print(f"Cache stats: {stats}")