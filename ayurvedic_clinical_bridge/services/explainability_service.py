"""
Explainability Service using SHAP for Ayurvedic Clinical Bridge
Optimized version with fast gradient-based explanations as primary method
"""

import shap
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path
import json
from transformers import AutoTokenizer
import time

logger = logging.getLogger(__name__)

class ExplainabilityService:
    """Service for providing fast explanations for model predictions."""
    
    def __init__(self, model, tokenizer, id_to_disease: Dict[str, str]):
        """
        Initialize the explainability service.
        
        Args:
            model: The trained PyTorch model
            tokenizer: The tokenizer used for preprocessing
            id_to_disease: Mapping from disease IDs to disease names
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id_to_disease = id_to_disease
        self.device = torch.device('cpu')
        
        # Initialize fast gradient-based explainer as primary method
        self._initialize_gradient_explainer()
        
        # Initialize SHAP as secondary method (for detailed analysis)
        self._initialize_shap_explainer()
        
        logger.info("Explainability service initialized successfully")
    
    def has_shap_explainer(self) -> bool:
        """Check if SHAP explainer is available."""
        return hasattr(self, 'explainer') and self.explainer is not None
    
    def get_explanation_methods(self) -> List[str]:
        """Get available explanation methods."""
        methods = []
        if hasattr(self, 'gradient_explainer') and self.gradient_explainer:
            methods.append("Gradient-based")
        if self.has_shap_explainer():
            methods.append("SHAP")
        return methods
    
    def _initialize_gradient_explainer(self):
        """Initialize fast gradient-based explainer."""
        try:
            def gradient_explainer(text, top_k=1):
                """Fast gradient-based explanation."""
                # Tokenize
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                )
                
                # Create embeddings that can require gradients
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                
                # Get embeddings from the model
                with torch.no_grad():
                    embeddings = self.model.bert.embeddings(input_ids)
                
                # Enable gradients on embeddings
                embeddings.requires_grad_(True)
                
                # Forward pass through the rest of the model
                outputs = self.model.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                logits = self.model.classifier(pooled_output)
                
                # Get predictions
                probs = torch.softmax(logits, dim=-1)
                top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                
                predictions = []
                for i in range(top_k):
                    disease_id = top_indices[0][i].item()
                    disease = self.id_to_disease[str(disease_id)]
                    confidence = top_probs[0][i].item()
                    
                    predictions.append({
                        'disease': disease,
                        'confidence': float(confidence),
                        'disease_id': disease_id
                    })
                
                # Backward pass for top prediction
                top_class = top_indices[0][0]
                logits[0, top_class].backward()
                
                # Get gradients from embeddings
                gradients = embeddings.grad
                
                # Calculate importance scores (sum over embedding dimensions)
                importance_scores = torch.sum(torch.abs(gradients), dim=-1).squeeze().cpu().numpy()
                
                # Get tokens
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                
                # Create word explanations
                word_explanations = []
                combined_words = []
                current_word = ""
                current_importance = 0.0
                
                for i, (token, score) in enumerate(zip(tokens, importance_scores)):
                    if token not in ['[CLS]', '[SEP]', '[PAD]']:
                        if token.startswith('##'):
                            # This is a subword token, combine with previous
                            current_word += token.replace('##', '')
                            current_importance += float(score)
                        else:
                            # This is a new word, save previous if exists
                            if current_word:
                                combined_words.append({
                                    'word': current_word,
                                    'importance': current_importance,
                                    'contribution': 'positive' if current_importance > 0.01 else 'negative' if current_importance < -0.01 else 'neutral'
                                })
                            
                            # Start new word
                            current_word = token
                            current_importance = float(score)
                
                # Don't forget the last word
                if current_word:
                    combined_words.append({
                        'word': current_word,
                        'importance': current_importance,
                        'contribution': 'positive' if current_importance > 0.01 else 'negative' if current_importance < -0.01 else 'neutral'
                    })
                
                # Sort by absolute importance
                combined_words.sort(key=lambda x: abs(x['importance']), reverse=True)
                
                return predictions, combined_words
            
            self.gradient_explainer = gradient_explainer
            logger.info("Fast gradient-based explainer initialized successfully")
            
        except Exception as e:
            logger.error(f"Gradient explainer initialization failed: {e}")
            self.gradient_explainer = None
    
    def _initialize_shap_explainer(self):
        """Initialize SHAP explainer as secondary method."""
        try:
            def model_wrapper(texts):
                """Wrapper function for SHAP to call the model."""
                if isinstance(texts, str):
                    texts = [texts]
                elif isinstance(texts, (list, np.ndarray)):
                    # Handle numpy array or list
                    if isinstance(texts, np.ndarray):
                        texts = texts.tolist()
                    # Ensure it's a list of strings
                    if texts and not isinstance(texts[0], str):
                        texts = [str(t) for t in texts]

                results = []
                for text in texts:
                    if not isinstance(text, str):
                        text = str(text)

                    inputs = self.tokenizer(
                        text,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=128
                    )

                    with torch.no_grad():
                        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                        logits = outputs['logits']
                        probs = torch.softmax(logits, dim=-1)
                        results.append(probs.cpu().numpy())

                result = np.vstack(results) if results else np.array([])
                return result

            # Initialize SHAP explainer with masking strategy
            try:
                masker = shap.maskers.Text(self.tokenizer)
                self.explainer = shap.Explainer(model_wrapper, masker)
                self.shap_available = True
                logger.info("SHAP explainer initialized successfully with Text masker")
            except TypeError as te:
                # Try without masker (simpler approach)
                try:
                    logger.warning(f"Text masker failed: {te}. Trying without masker...")
                    self.explainer = shap.Explainer(model_wrapper)
                    self.shap_available = True
                    logger.info("SHAP explainer initialized without masker")
                except Exception as e2:
                    logger.warning(f"SHAP initialization failed completely: {e2}. Using gradient method only.")
                    self.explainer = None
                    self.shap_available = False
            except Exception as shap_init_error:
                logger.warning(f"SHAP Explainer initialization failed: {shap_init_error}. Continuing with gradient-based explanations only.")
                self.explainer = None
                self.shap_available = False

        except Exception as e:
            logger.warning(f"SHAP explainer initialization deferred: {e}")
            self.explainer = None
            self.shap_available = False
    
    def explain_prediction(self, text: str, top_k: int = 5, use_shap: bool = False) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.
        
        Args:
            text: Input text to explain
            top_k: Number of top predictions to explain
            use_shap: Whether to use SHAP (slower) or gradient method (faster)
            
        Returns:
            Dictionary containing explanation data
        """
        try:
            start_time = time.time()
            
            # Use SHAP if requested and available
            if use_shap and self.explainer:
                try:
                    # Get SHAP explanation
                    shap_values = self.explainer([text])
                    
                    # Get model predictions
                    inputs = self.tokenizer(
                        text, 
                        return_tensors='pt', 
                        padding=True, 
                        truncation=True, 
                        max_length=128
                    )
                    
                    with torch.no_grad():
                        outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                        logits = outputs['logits']
                        probs = torch.softmax(logits, dim=-1)
                        top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                    
                    predictions = []
                    for i in range(top_k):
                        disease_id = top_indices[0][i].item()
                        disease = self.id_to_disease[str(disease_id)]
                        confidence = top_probs[0][i].item()
                        
                        predictions.append({
                            'disease': disease,
                            'confidence': float(confidence),
                            'disease_id': disease_id
                        })
                    
                    # Extract word explanations from SHAP values
                    word_explanations = []
                    if hasattr(shap_values, 'data') and hasattr(shap_values, 'values'):
                        words = shap_values.data[0]
                        values = shap_values.values[0]
                        
                        # Get top prediction class values
                        top_class_idx = top_indices[0][0].item()
                        if len(values.shape) > 1 and top_class_idx < values.shape[1]:
                            class_values = values[:, top_class_idx]
                        else:
                            class_values = values
                        
                        for word, value in zip(words, class_values):
                            if word.strip() and word not in ['[CLS]', '[SEP]', '[PAD]']:
                                word_explanations.append({
                                    'word': word.strip(),
                                    'importance': float(value),
                                    'contribution': 'positive' if value > 0.001 else 'negative' if value < -0.001 else 'neutral'
                                })
                    
                    # Sort by absolute importance
                    word_explanations.sort(key=lambda x: abs(x['importance']), reverse=True)
                    explanation_method = "SHAP (SHapley Additive exPlanations)"
                    
                except Exception as shap_error:
                    logger.warning(f"SHAP explanation failed, falling back to gradient method: {shap_error}")
                    # Fall back to gradient method
                    predictions, word_explanations = self.gradient_explainer(text, top_k)
                    explanation_method = "Gradient-based (SHAP fallback)"
            
            # Use fast gradient-based method by default
            elif self.gradient_explainer:
                predictions, word_explanations = self.gradient_explainer(text, top_k)
                explanation_method = "Gradient-based (Fast)"
                
            else:
                # Fallback to basic prediction without detailed explanation
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True, 
                    max_length=128
                )
                
                with torch.no_grad():
                    outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
                    logits = outputs['logits']
                    probs = torch.softmax(logits, dim=-1)
                    
                    # Get top predictions
                    top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
                    
                    predictions = []
                    for i in range(top_k):
                        disease_id = top_indices[0][i].item()
                        disease = self.id_to_disease[str(disease_id)]
                        confidence = top_probs[0][i].item()
                        
                        predictions.append({
                            'disease': disease,
                            'confidence': float(confidence),
                            'disease_id': disease_id
                        })
                
                word_explanations = []
                explanation_method = "Basic prediction (no detailed explanation)"
            
            processing_time = time.time() - start_time
            
            # Generate summary explanation
            summary = self._generate_explanation_summary(
                text, predictions[0] if predictions else {}, word_explanations[:10]
            )
            
            # Create visualization data
            visualization_data = self._create_visualization_data(
                word_explanations[:15], predictions[0] if predictions else {}
            )
            
            return {
                'explanation_available': True,
                'input_text': text,
                'top_prediction': predictions[0] if predictions else {},
                'all_predictions': predictions,
                'word_explanations': word_explanations,
                'summary': summary,
                'visualization_data': visualization_data,
                'explanation_method': explanation_method,
                'processing_time': processing_time,
                'confidence_threshold': 0.01,
                'important_words_count': len([w for w in word_explanations if abs(w['importance']) > 0.01])
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return {
                'error': f'Failed to generate explanation: {str(e)}',
                'explanation_available': False
            }
    
    def _generate_explanation_summary(self, text: str, prediction: Dict, 
                                    top_words: List[Dict]) -> str:
        """Generate a human-readable explanation summary."""
        if not prediction:
            return "Unable to generate prediction explanation."
            
        disease = prediction.get('disease', 'Unknown')
        confidence = prediction.get('confidence', 0) * 100
        
        summary = f"The AI model predicted '{disease}' with {confidence:.1f}% confidence. "
        
        if top_words:
            positive_words = [w['word'] for w in top_words if w['contribution'] == 'positive'][:3]
            
            if positive_words:
                summary += f"Key words that influenced this prediction: {', '.join(positive_words)}. "
        
        summary += "This explanation shows which words in your input were most important for the AI's decision."
        
        return summary
    
    def _create_visualization_data(self, word_explanations: List[Dict], 
                                 prediction: Dict) -> Dict[str, Any]:
        """Create data for visualization components."""
        return {
            'words': [w['word'] for w in word_explanations],
            'importance_scores': [w['importance'] for w in word_explanations],
            'contributions': [w['contribution'] for w in word_explanations],
            'prediction_info': {
                'disease': prediction.get('disease', 'Unknown'),
                'confidence': prediction.get('confidence', 0)
            },
            'color_scale': {
                'positive': '#2E8B57',  # Sea Green
                'negative': '#DC143C',  # Crimson
                'neutral': '#808080'    # Gray
            }
        }
    
    def explain_multiple_predictions(self, texts: List[str], 
                                   top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple predictions.
        
        Args:
            texts: List of input texts to explain
            top_k: Number of top predictions to explain for each
            
        Returns:
            List of explanation dictionaries
        """
        explanations = []
        
        for text in texts:
            explanation = self.explain_prediction(text, top_k, use_shap=False)  # Use fast method
            explanations.append(explanation)
        
        return explanations
    
    def get_global_feature_importance(self, sample_texts: List[str] = None) -> Dict[str, Any]:
        """
        Get global feature importance across multiple samples.
        
        Args:
            sample_texts: Sample texts to analyze (optional)
            
        Returns:
            Global feature importance data
        """
        if not sample_texts:
            # Use default sample texts representing different conditions
            sample_texts = [
                "fever headache body ache",
                "cough cold runny nose",
                "stomach pain nausea vomiting",
                "joint pain swelling stiffness",
                "chest pain breathing difficulty"
            ]
        
        try:
            # Get explanations for all samples using fast method
            all_explanations = []
            for text in sample_texts:
                explanation = self.explain_prediction(text, top_k=1, use_shap=False)
                if explanation.get('explanation_available'):
                    all_explanations.append(explanation)
            
            if not all_explanations:
                return {'error': 'No valid explanations generated'}
            
            # Aggregate word importance across all samples
            word_importance = {}
            word_counts = {}
            
            for explanation in all_explanations:
                for word_data in explanation.get('word_explanations', []):
                    word = word_data['word']
                    importance = abs(word_data['importance'])
                    
                    if word not in word_importance:
                        word_importance[word] = 0
                        word_counts[word] = 0
                    
                    word_importance[word] += importance
                    word_counts[word] += 1
            
            # Calculate average importance
            avg_importance = {}
            for word in word_importance:
                avg_importance[word] = word_importance[word] / word_counts[word]
            
            # Sort by importance
            sorted_words = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            # Get top important words
            top_global_words = []
            for word, importance in sorted_words[:20]:
                top_global_words.append({
                    'word': word,
                    'average_importance': importance,
                    'frequency': word_counts[word],
                    'total_importance': word_importance[word]
                })
            
            return {
                'global_importance_available': True,
                'top_important_words': top_global_words,
                'total_samples_analyzed': len(all_explanations),
                'unique_words_found': len(word_importance),
                'analysis_method': 'Aggregated gradient-based importance across multiple samples'
            }
            
        except Exception as e:
            logger.error(f"Error generating global feature importance: {e}")
            return {
                'error': f'Failed to generate global importance: {str(e)}',
                'global_importance_available': False
            }
