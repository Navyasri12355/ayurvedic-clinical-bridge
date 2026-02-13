"""
BiLSTM-CRF Clinical Processor

This module provides a clinical processor using BiLSTM-CRF for fast and accurate
entity recognition, designed to complement the existing BioBERT processor.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import logging
import time
from pathlib import Path
import json

from .bilstm_crf_model import BiLSTMCRF, BiLSTMCRFConfig, BiLSTMCRFTokenizer, create_default_vocab
from .biobert_transformer import ClinicalEntityType, TaskType, ClinicalPrediction

logger = logging.getLogger(__name__)


class BiLSTMCRFClinicalProcessor:
    """
    Clinical processor using BiLSTM-CRF for fast entity recognition.
    
    This processor is optimized for speed while maintaining good accuracy,
    making it suitable for general users and real-time applications.
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[BiLSTMCRFConfig] = None):
        """Initialize the BiLSTM-CRF clinical processor."""
        self.config = config or BiLSTMCRFConfig()
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Entity type mappings (same as BioBERT for consistency)
        self.id_to_label = {entity.value: entity.name for entity in ClinicalEntityType}
        self.label_to_id = {entity.name: entity.value for entity in ClinicalEntityType}
        
        # Try to load trained model
        if model_path:
            self.load_model(model_path)
        else:
            # Try default paths
            model_paths = [
                Path("models/bilstm_crf"),
                Path("models/bilstm_crf/best_model.pt")
            ]
            
            for path in model_paths:
                if path.exists():
                    try:
                        self.load_model(str(path))
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load model from {path}: {e}")
                        continue
        
        # If no model loaded, initialize with default configuration
        if self.model is None:
            logger.info("No trained BiLSTM-CRF model found, initializing with default configuration")
            self._initialize_default_model()
        
        logger.info("BiLSTM-CRF clinical processor initialized")
    
    def _initialize_default_model(self):
        """Initialize model with default configuration for demonstration."""
        # Create default vocabulary
        vocab_to_idx = create_default_vocab()
        self.config.vocab_size = len(vocab_to_idx)
        
        # Initialize tokenizer
        self.tokenizer = BiLSTMCRFTokenizer(vocab_to_idx)
        
        # Initialize model
        self.model = BiLSTMCRF(self.config, vocab_to_idx)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Initialized default BiLSTM-CRF model with {len(vocab_to_idx)} vocabulary size")
    
    def load_model(self, model_path: str):
        """Load trained BiLSTM-CRF model from directory or file."""
        try:
            model_path = Path(model_path)
            
            if model_path.is_file() and model_path.suffix == '.pt':
                # Load from single file
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Extract configuration and vocabulary
                if 'config' in checkpoint:
                    config_dict = checkpoint['config']
                    self.config = BiLSTMCRFConfig(**config_dict)
                
                vocab_to_idx = checkpoint.get('vocab_to_idx', create_default_vocab())
                self.config.vocab_size = len(vocab_to_idx)
                
                # Initialize tokenizer
                self.tokenizer = BiLSTMCRFTokenizer(vocab_to_idx)
                
                # Initialize and load model
                self.model = BiLSTMCRF(self.config, vocab_to_idx)
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Loaded BiLSTM-CRF model from {model_path}")
                
            elif model_path.is_dir():
                # Load from directory
                config_path = model_path / "config.json"
                model_file = model_path / "pytorch_model.bin"
                vocab_path = model_path / "vocab.json"
                
                # Load configuration
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config_dict = json.load(f)
                    self.config = BiLSTMCRFConfig(**config_dict)
                
                # Load vocabulary
                if vocab_path.exists():
                    with open(vocab_path, 'r') as f:
                        vocab_to_idx = json.load(f)
                else:
                    vocab_to_idx = create_default_vocab()
                
                self.config.vocab_size = len(vocab_to_idx)
                
                # Initialize tokenizer
                self.tokenizer = BiLSTMCRFTokenizer(vocab_to_idx)
                
                # Initialize and load model
                self.model = BiLSTMCRF(self.config, vocab_to_idx)
                
                if model_file.exists():
                    checkpoint = torch.load(model_file, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                
                logger.info(f"Loaded BiLSTM-CRF model from directory {model_path}")
            
            else:
                raise FileNotFoundError(f"Model path {model_path} not found")
                
        except Exception as e:
            logger.error(f"Failed to load BiLSTM-CRF model: {e}")
            # Fall back to default initialization
            self._initialize_default_model()
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract clinical entities from text using BiLSTM-CRF.
        
        Args:
            text: Input clinical text
            
        Returns:
            List of extracted entities with positions and confidence scores
        """
        if not self.model or not self.tokenizer:
            logger.warning("BiLSTM-CRF model not loaded, using fallback entity extraction")
            return self._fallback_entity_extraction(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                text,
                max_length=self.config.max_length,
                padding=True,
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs["predictions"]
            
            # Convert predictions to entities
            if predictions.dim() > 1:
                predictions = predictions[0].cpu().numpy()  # Take first batch item
            else:
                predictions = predictions.cpu().numpy()
            
            # Get original tokens for alignment
            input_ids = inputs['input_ids'][0].cpu().numpy()
            attention_mask = inputs['attention_mask'][0].cpu().numpy()
            
            # Decode tokens to get word boundaries
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids.tolist())
            
            entities = []
            current_entity = None
            current_pos = 0
            
            for i, (token, pred_id, mask) in enumerate(zip(tokens, predictions, attention_mask)):
                if not mask or token in ['[PAD]', '[CLS]', '[SEP]']:
                    continue
                
                # Find token position in original text
                token_start = text.lower().find(token, current_pos)
                if token_start == -1:
                    # Skip if token not found (might be subword)
                    continue
                
                token_end = token_start + len(token)
                current_pos = token_end
                
                # Safe entity type lookup
                try:
                    pred_id = int(pred_id)
                    if pred_id in self.id_to_label:
                        label = self.id_to_label[pred_id]
                    else:
                        label = "O"
                except (ValueError, KeyError):
                    label = "O"
                
                if label.startswith('B_'):
                    # Start of new entity
                    if current_entity:
                        entities.append(current_entity)
                    
                    entity_type = label[2:]  # Remove 'B_' prefix
                    current_entity = {
                        'type': entity_type,
                        'text': text[token_start:token_end],
                        'start': token_start,
                        'end': token_end,
                        'confidence': 0.85  # BiLSTM-CRF confidence
                    }
                
                elif label.startswith('I_') and current_entity:
                    # Continue current entity
                    entity_type = label[2:]  # Remove 'I_' prefix
                    if current_entity['type'] == entity_type:
                        current_entity['text'] = text[current_entity['start']:token_end]
                        current_entity['end'] = token_end
                
                else:
                    # Outside or different entity type
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # Add final entity if exists
            if current_entity:
                entities.append(current_entity)
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in BiLSTM-CRF entity extraction: {e}")
            return self._fallback_entity_extraction(text)
    
    def _fallback_entity_extraction(self, text: str) -> List[Dict[str, Any]]:
        """Fallback entity extraction using simple rules."""
        entities = []
        text_lower = text.lower()
        
        # Enhanced patterns for better coverage
        entity_patterns = {
            'HERB': [
                'turmeric', 'ginger', 'ashwagandha', 'brahmi', 'gudmar', 'arjuna',
                'tulsi', 'neem', 'amla', 'triphala', 'guggulu', 'shatavari',
                'licorice', 'fenugreek', 'cinnamon', 'cardamom', 'cumin',
                'gymnema sylvestre', 'terminalia arjuna', 'withania somnifera',
                'curcuma longa', 'zingiber officinale', 'ocimum sanctum'
            ],
            'DISEASE': [
                'diabetes', 'type 1 diabetes', 'type 2 diabetes', 'diabetes mellitus',
                'hypertension', 'high blood pressure', 'arthritis', 'asthma', 'migraine',
                'anxiety', 'depression', 'insomnia', 'obesity', 'fever',
                'cough', 'cold', 'indigestion', 'constipation', 'acidity',
                'heart disease', 'kidney disease', 'liver disease', 'gastritis'
            ],
            'DRUG': [
                'metformin', 'insulin', 'aspirin', 'lisinopril', 'atorvastatin',
                'warfarin', 'ibuprofen', 'acetaminophen', 'paracetamol', 'prednisone'
            ],
            'SYMPTOM': [
                'pain', 'inflammation', 'swelling', 'headache', 'nausea',
                'fatigue', 'dizziness', 'burning', 'stiffness', 'weakness',
                'thirst', 'frequent urination', 'blurred vision', 'chest pain',
                'shortness of breath', 'joint pain', 'muscle pain'
            ],
            'DOSAGE': [
                'mg', 'gram', 'grams', 'tsp', 'teaspoon', 'tbsp', 'tablespoon',
                'daily', 'twice daily', 'three times', 'capsule', 'tablet',
                '500mg', '250mg', '1000mg', 'ml', 'drops'
            ]
        }
        
        for entity_type, patterns in entity_patterns.items():
            for pattern in patterns:
                start_idx = 0
                while True:
                    idx = text_lower.find(pattern, start_idx)
                    if idx == -1:
                        break
                    
                    # Check word boundaries
                    if (idx == 0 or not text[idx-1].isalnum()) and \
                       (idx + len(pattern) == len(text) or not text[idx + len(pattern)].isalnum()):
                        entities.append({
                            'type': entity_type,
                            'text': text[idx:idx+len(pattern)],
                            'start': idx,
                            'end': idx + len(pattern),
                            'confidence': 0.75  # Lower confidence for rule-based
                        })
                    
                    start_idx = idx + 1
        
        # Remove overlaps, keeping longest matches
        unique_entities = []
        entities_sorted = sorted(entities, key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        for entity in entities_sorted:
            overlaps = False
            for existing in unique_entities:
                if (entity['start'] < existing['end'] and entity['end'] > existing['start']):
                    overlaps = True
                    break
            
            if not overlaps:
                unique_entities.append(entity)
        
        return sorted(unique_entities, key=lambda x: x['start'])
    
    def detect_interactions(self, herbs: List[str], drugs: List[str]) -> List[Dict[str, Any]]:
        """
        Detect potential herb-drug interactions.
        
        This is a simplified version compared to BioBERT, focusing on speed.
        """
        interactions = []
        
        # Known interaction patterns (simplified for speed)
        known_interactions = {
            ('turmeric', 'warfarin'): {
                'severity': 'high',
                'mechanism': 'Increased bleeding risk',
                'confidence': 0.9
            },
            ('ginger', 'warfarin'): {
                'severity': 'moderate',
                'mechanism': 'Anticoagulant effects',
                'confidence': 0.8
            },
            ('licorice', 'lisinopril'): {
                'severity': 'moderate',
                'mechanism': 'Potassium depletion',
                'confidence': 0.75
            }
        }
        
        for herb in herbs:
            for drug in drugs:
                herb_lower = herb.lower()
                drug_lower = drug.lower()
                
                # Check known interactions
                interaction_key = (herb_lower, drug_lower)
                if interaction_key in known_interactions:
                    interaction_data = known_interactions[interaction_key]
                    interactions.append({
                        'herb': herb,
                        'drug': drug,
                        'severity': interaction_data['severity'],
                        'mechanism': interaction_data['mechanism'],
                        'confidence': interaction_data['confidence'],
                        'evidence_level': 'literature_based'
                    })
        
        return interactions
    
    def assess_safety(self, entities: List[Dict[str, Any]], 
                     interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform safety assessment (simplified version for speed).
        """
        drugs = [e for e in entities if e['type'] == 'DRUG']
        herbs = [e for e in entities if e['type'] == 'HERB']
        diseases = [e for e in entities if e['type'] == 'DISEASE']
        
        # Quick risk calculation
        risk_factors = []
        
        # Interaction risk
        high_risk_interactions = [i for i in interactions if i.get('severity') == 'high']
        moderate_risk_interactions = [i for i in interactions if i.get('severity') == 'moderate']
        
        interaction_risk = len(high_risk_interactions) * 0.8 + len(moderate_risk_interactions) * 0.4
        risk_factors.append(min(interaction_risk, 1.0))
        
        # Polypharmacy risk
        polypharmacy_risk = min(len(drugs) * 0.1, 0.5)
        risk_factors.append(polypharmacy_risk)
        
        overall_risk = min(sum(risk_factors), 1.0)
        
        recommendations = []
        if overall_risk > 0.7:
            recommendations.append("High risk detected. Consult healthcare provider.")
        elif overall_risk > 0.4:
            recommendations.append("Moderate risk. Monitor closely.")
        else:
            recommendations.append("Low risk. Standard monitoring recommended.")
        
        return {
            'overall_risk_score': overall_risk,
            'risk_level': 'high' if overall_risk > 0.7 else 'moderate' if overall_risk > 0.4 else 'low',
            'risk_factors': {
                'interaction_risk': interaction_risk,
                'polypharmacy_risk': polypharmacy_risk
            },
            'recommendations': recommendations,
            'requires_consultation': overall_risk > 0.6
        }
    
    def recommend_treatments(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Recommend treatments based on entities (simplified for speed).
        """
        diseases = [e for e in entities if e['type'] == 'DISEASE']
        symptoms = [e for e in entities if e['type'] == 'SYMPTOM']
        
        recommendations = []
        
        # Quick treatment recommendations
        treatment_map = {
            'diabetes': {
                'herb': 'Gymnema sylvestre (Gudmar)',
                'dosage': '500mg twice daily',
                'mechanism': 'Glucose metabolism support',
                'confidence': 0.8
            },
            'hypertension': {
                'herb': 'Terminalia arjuna',
                'dosage': '500mg three times daily',
                'mechanism': 'Cardiovascular support',
                'confidence': 0.75
            },
            'arthritis': {
                'herb': 'Turmeric (Curcuma longa)',
                'dosage': '1000mg twice daily',
                'mechanism': 'Anti-inflammatory action',
                'confidence': 0.85
            }
        }
        
        for disease in diseases:
            disease_name = disease['text'].lower()
            for key, treatment in treatment_map.items():
                if key in disease_name:
                    recommendations.append({
                        'condition': disease['text'],
                        'herb': treatment['herb'],
                        'dosage': treatment['dosage'],
                        'mechanism': treatment['mechanism'],
                        'confidence': treatment['confidence'],
                        'evidence_level': 'traditional_use'
                    })
                    break
        
        return recommendations
    
    def process_clinical_text(self, text: str, task_types: List[TaskType] = None) -> ClinicalPrediction:
        """
        Process clinical text using BiLSTM-CRF (optimized for speed).
        """
        start_time = time.time()
        
        if task_types is None:
            task_types = list(TaskType)
        
        # Initialize results
        entities = []
        interactions = []
        safety_assessment = {}
        treatment_recommendations = []
        confidence_scores = {}
        
        # Entity Recognition (always performed for BiLSTM-CRF)
        if TaskType.ENTITY_RECOGNITION in task_types:
            entities = self.extract_entities(text)
            confidence_scores['entity_recognition'] = 0.85
        
        # Interaction Detection
        if TaskType.INTERACTION_DETECTION in task_types:
            herbs = [e['text'] for e in entities if e['type'] == 'HERB']
            drugs = [e['text'] for e in entities if e['type'] == 'DRUG']
            interactions = self.detect_interactions(herbs, drugs)
            confidence_scores['interaction_detection'] = 0.75
        
        # Safety Assessment
        if TaskType.SAFETY_ASSESSMENT in task_types:
            safety_assessment = self.assess_safety(entities, interactions)
            confidence_scores['safety_assessment'] = 0.8
        
        # Treatment Recommendations
        if TaskType.TREATMENT_RECOMMENDATION in task_types:
            treatment_recommendations = self.recommend_treatments(entities)
            confidence_scores['treatment_recommendation'] = 0.7
        
        processing_time = time.time() - start_time
        
        return ClinicalPrediction(
            text=text,
            entities=entities,
            interactions=interactions,
            safety_assessment=safety_assessment,
            treatment_recommendations=treatment_recommendations,
            confidence_scores=confidence_scores,
            processing_time=processing_time,
            metadata={
                'model_type': 'BiLSTM-CRF',
                'task_types': [t.value for t in task_types],
                'num_entities': len(entities),
                'num_interactions': len(interactions),
                'optimized_for': 'speed'
            }
        )
    
    def is_available(self) -> bool:
        """Check if the BiLSTM-CRF model is available."""
        return self.model is not None and self.tokenizer is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_available():
            return {'available': False}
        
        return {
            'available': True,
            'model_type': 'BiLSTM-CRF',
            'vocab_size': self.config.vocab_size,
            'embedding_dim': self.config.embedding_dim,
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'num_labels': self.config.num_labels,
            'parameters': self.model._count_parameters() if self.model else 0,
            'device': str(self.device),
            'optimized_for': 'speed_and_accuracy'
        }


# Global processor instance
_bilstm_crf_processor = None

def get_bilstm_crf_processor() -> BiLSTMCRFClinicalProcessor:
    """Get the global BiLSTM-CRF clinical processor instance."""
    global _bilstm_crf_processor
    if _bilstm_crf_processor is None:
        _bilstm_crf_processor = BiLSTMCRFClinicalProcessor()
    return _bilstm_crf_processor