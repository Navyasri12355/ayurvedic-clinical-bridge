"""
Herb Predictor Service
Pure ML predictor for herb benefits, side effects, and traditional properties
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import logging

logger = logging.getLogger(__name__)


class HerbBioBERT(nn.Module):
    """BioBERT model for herb classification tasks."""
    
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes
        
        # Load BioBERT
        self.bert = AutoModel.from_pretrained(config['biobert_model'])
        
        # Classification head
        hidden_size = self.bert.config.hidden_size
        if config.get('simple', False) or 'simple' in str(config).lower():
            # Simple classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            # Complex classifier
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_classes)
            )
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return {'logits': logits}


class HerbPredictor:
    """Pure ML predictor for herb-related queries."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.models = {}
        self.tokenizers = {}
        self.mappings = {}
        self.herb_data = []
        self.synonym_matcher = None
        self._load_herb_data()
        self._initialize_synonym_matcher()
        self.load_models()
    
    def find_herbs_for_symptom(self, symptom: str) -> list:
        """
        Find herbs that are known to treat a specific symptom/condition.
        Searches through herb data reviews and traditional properties.
        """
        relevant_herbs = []
        symptom_lower = symptom.lower()
        
        # Keywords to check in 'prabhav' (actions) or preview text
        # Simple string matching for now
        for herb in self.herb_data:
            score = 0
            
            # Check preview text
            if 'preview' in herb and herb['preview']:
                if symptom_lower in herb['preview'].lower():
                    score += 2
            
            # Check properties/actions
            if 'prabhav' in herb and herb['prabhav']:
                # prabhav is a list of strings
                for action in herb['prabhav']:
                    if symptom_lower in action.lower():
                        score += 3
            
            if score > 0:
                relevant_herbs.append({
                    'name': herb['name'],
                    'score': score,
                    'reason': f"Matches symptom '{symptom}'"
                })
        
        # Sort by score descending
        relevant_herbs.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top 3 unique herb names
        return [h['name'] for h in relevant_herbs[:3]]
    
    def _initialize_synonym_matcher(self):
        """Initialize the herb synonym matcher."""
        try:
            from .herb_synonym_matcher import HerbSynonymMatcher
            self.synonym_matcher = HerbSynonymMatcher()
            logger.info("Herb synonym matcher initialized")
        except Exception as e:
            logger.error(f"Error initializing synonym matcher: {e}")
            self.synonym_matcher = None

    def load_models(self):
        """Load all available herb models."""
        model_types = ['benefits']  # Start with benefits model
        
        for model_type in model_types:
            model_dir = Path(f"models/herb_{model_type}")
            
            if model_dir.exists():
                try:
                    logger.info(f"Loading herb {model_type} model...")
                    
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(model_dir)
                    
                    # Load model checkpoint
                    checkpoint = torch.load(model_dir / "pytorch_model.bin", map_location=self.device)
                    
                    # Load mappings
                    mapping_file = model_dir / f"{model_type}_mappings.json"
                    if not mapping_file.exists():
                        # Try alternative naming
                        mapping_file = model_dir / f"{model_type.rstrip('s')}_mappings.json"
                    
                    if mapping_file.exists():
                        with open(mapping_file, 'r') as f:
                            mappings = json.load(f)
                        
                        # Ensure we have id_to_class mapping
                        if 'benefit_to_id' in mappings and 'id_to_class' not in mappings:
                            if 'id_to_benefit' in mappings:
                                mappings['id_to_class'] = mappings['id_to_benefit']
                            else:
                                # Create reverse mapping
                                mappings['id_to_class'] = {str(v): k for k, v in mappings['benefit_to_id'].items()}
                        elif 'class_to_id' in mappings and 'id_to_class' not in mappings:
                            # Create reverse mapping
                            mappings['id_to_class'] = {str(v): k for k, v in mappings['class_to_id'].items()}
                    else:
                        logger.warning(f"No mappings found for {model_type} model")
                        continue
                    
                    # Create model instance
                    config = {
                        'biobert_model': 'dmis-lab/biobert-base-cased-v1.1',
                        'simple': True
                    }
                    
                    # Get number of classes from mappings
                    if 'num_classes' in mappings:
                        num_classes = mappings['num_classes']
                    elif 'class_to_id' in mappings:
                        num_classes = len(mappings['class_to_id'])
                    elif 'benefit_to_id' in mappings:
                        num_classes = len(mappings['benefit_to_id'])
                    elif 'id_to_class' in mappings:
                        num_classes = len(mappings['id_to_class'])
                    else:
                        logger.error(f"Could not determine number of classes for {model_type} model")
                        continue
                    
                    model = HerbBioBERT(config, num_classes)
                    
                    # Load state dict
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                    
                    model.eval()
                    
                    # Store components
                    self.models[model_type] = model
                    self.tokenizers[model_type] = tokenizer
                    self.mappings[model_type] = mappings
                    
                    logger.info(f"Successfully loaded herb {model_type} model")
                    
                except Exception as e:
                    logger.error(f"Failed to load herb {model_type} model: {e}")
                    continue
        
        logger.info(f"Loaded {len(self.models)} herb models: {list(self.models.keys())}")
    
    def predict_herb_benefits(self, herb_name, top_k=5):
        """Predict benefits for a given herb."""
        if 'benefits' not in self.models:
            logger.warning("Benefits model not available")
            return []
        
        try:
            model = self.models['benefits']
            tokenizer = self.tokenizers['benefits']
            mappings = self.mappings['benefits']
            
            # Tokenize input
            inputs = tokenizer(
                herb_name,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            # Remove token_type_ids if present (not needed for BioBERT)
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = outputs['logits']
            
            # Get probabilities
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, min(top_k, len(mappings['id_to_class'])), dim=-1)
            
            # Format results
            predictions = []
            for i in range(len(top_indices[0])):
                class_id = top_indices[0][i].item()
                class_name = mappings['id_to_class'].get(str(class_id), f"Unknown_{class_id}")
                confidence = top_probs[0][i].item()
                
                predictions.append({
                    'benefit': class_name,
                    'confidence': float(confidence),
                    'confidence_percentage': f"{confidence * 100:.1f}%"
                })
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting herb benefits: {e}")
            return []
    
    def get_herb_information(self, herb_name):
        """Get comprehensive herb information with synonym matching."""
        # First try to resolve the herb name using synonym matching
        resolved_name = herb_name
        if self.synonym_matcher:
            resolved_name = self.synonym_matcher.resolve_herb_name(herb_name)
            if resolved_name:
                logger.info(f"Resolved '{herb_name}' to '{resolved_name}' using synonym matching")
            else:
                # If no exact match, get suggestions
                suggestions = self.synonym_matcher.get_herb_suggestions(herb_name, max_suggestions=3)
                if suggestions:
                    logger.info(f"Found {len(suggestions)} suggestions for '{herb_name}': {[s['name'] for s in suggestions]}")
                    # Use the best suggestion if confidence is high enough
                    if suggestions[0]['confidence'] >= 0.8:
                        resolved_name = suggestions[0]['canonical_name']
                        logger.info(f"Using best suggestion '{resolved_name}' for '{herb_name}'")
        
        # Find herb in data using resolved name
        herb_info = None
        search_name = resolved_name if resolved_name else herb_name
        
        for herb in self.herb_data:
            if herb['name'].lower() == search_name.lower():
                herb_info = herb
                break
        
        if not herb_info:
            # If still not found, try original name as fallback
            for herb in self.herb_data:
                if herb['name'].lower() == herb_name.lower():
                    herb_info = herb
                    break
        
        if not herb_info:
            # Return suggestions if available
            suggestions = []
            if self.synonym_matcher:
                suggestions = self.synonym_matcher.get_herb_suggestions(herb_name, max_suggestions=5)
            
            return {
                'name': herb_name,
                'found': False,
                'message': f"Herb '{herb_name}' not found in database",
                'suggestions': suggestions,
                'search_attempted': resolved_name if resolved_name != herb_name else None
            }
        
        # Format comprehensive information
        return {
            'name': herb_info['name'],
            'found': True,
            'preview': herb_info.get('preview', ''),
            'traditional_properties': {
                'rasa': herb_info.get('rasa', []),
                'guna': herb_info.get('guna', []),
                'virya': herb_info.get('virya', ''),
                'vipaka': herb_info.get('vipaka', ''),
                'prabhav': herb_info.get('prabhav', [])
            },
            'dosha_effects': {
                'pacifies': herb_info.get('pacify', []),
                'aggravates': herb_info.get('aggravate', []),
                'tridoshic': herb_info.get('tridosha', False)
            },
            'link': herb_info.get('link', '')
        }
    
    def is_available(self):
        """Check if herb predictor is available."""
        return len(self.models) > 0
    
    def get_available_models(self):
        """Get list of available models."""
        return list(self.models.keys())