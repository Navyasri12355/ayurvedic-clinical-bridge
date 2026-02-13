#!/usr/bin/env python3
"""
Query Intent Classifier Service
Classifies queries into herb_benefits, disease_prediction, or general_info
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class QueryIntentClassifier(nn.Module):
    """BERT-based query intent classifier."""
    
    def __init__(self, model_name='dmis-lab/biobert-v1.1', num_classes=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes = num_classes
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
        return {'loss': loss, 'logits': logits} if loss is not None else logits

class QueryIntentClassifierService:
    """Service for classifying query intents using trained ML model."""
    
    def __init__(self):
        """Initialize the query intent classifier service."""
        self.model = None
        self.tokenizer = None
        self.label_mapping = {}
        self.is_loaded = False
        
        self.load_model()
    
    def load_model(self):
        """Load the trained query intent classifier model."""
        try:
            model_path = Path('models/query_intent_classifier')
            
            if not model_path.exists():
                logger.warning("Query intent classifier model not found. Using fallback classification.")
                return
            
            # Load label mapping
            label_mapping_path = model_path.parent / 'query_intent_label_mapping.json'
            if label_mapping_path.exists():
                with open(label_mapping_path, 'r') as f:
                    self.label_mapping = json.load(f)
            else:
                # Default mapping
                self.label_mapping = {
                    '0': 'disease_prediction',
                    '1': 'general_info', 
                    '2': 'herb_benefits'
                }
            
            # Load config
            config_path = model_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    num_classes = config.get('num_classes', 3)
                    model_name = config.get('model_name', 'dmis-lab/biobert-v1.1')
            else:
                num_classes = 3
                model_name = 'dmis-lab/biobert-v1.1'
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load model
            self.model = QueryIntentClassifier(model_name, num_classes)
            
            # Load model weights
            model_weights_path = model_path / 'pytorch_model.bin'
            if model_weights_path.exists():
                self.model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
                self.model.eval()
                self.is_loaded = True
                logger.info("Query intent classifier loaded successfully")
            else:
                logger.warning("Model weights not found")
                
        except Exception as e:
            logger.error(f"Failed to load query intent classifier: {e}")
            self.model = None
            self.tokenizer = None
            self.is_loaded = False
    
    def classify_intent(self, query: str) -> tuple[str, float]:
        """
        Classify query intent using the trained ML model with herb-aware enhancement.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (intent, confidence)
        """
        # First, check for obvious herb patterns before using ML model
        query_lower = query.lower()
        
        # Enhanced herb detection
        herb_benefit_patterns = [
            'benefits of', 'uses of', 'properties of', 'effects of',
            'advantages of', 'good for', 'help with', 'medicinal properties',
            'therapeutic effects', 'healing properties'
        ]
        
        common_herbs = [
            'turmeric', 'haridra', 'ginger', 'adrak', 'ashwagandha', 'tulsi', 'holy basil',
            'neem', 'nimba', 'brahmi', 'bacopa', 'triphala', 'amla', 'amalaki',
            'guduchi', 'giloy', 'shankhpushpi', 'arjuna', 'fenugreek', 'methi'
        ]
        
        # If query clearly matches herb benefit patterns, return herb_benefits with high confidence
        contains_herb = any(herb in query_lower for herb in common_herbs)
        contains_benefit_pattern = any(pattern in query_lower for pattern in herb_benefit_patterns)
        
        if contains_herb and contains_benefit_pattern:
            return 'herb_benefits', 0.95
        
        # If not clearly a herb query, use ML model
        if not self.is_loaded or not self.model or not self.tokenizer:
            # Fallback to simple rule-based classification
            return self._fallback_classification(query)
        
        try:
            # Tokenize input
            encoding = self.tokenizer(
                query,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(encoding['input_ids'], encoding['attention_mask'])
                
                # Handle both dict and tensor outputs
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Get probabilities and prediction
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities.max().item()
                
                # Map prediction to label
                intent = self.label_mapping.get(str(prediction), 'general_info')
                
                # Post-process: if ML says general_info but we detected herb patterns, override
                if intent == 'general_info' and (contains_herb or contains_benefit_pattern):
                    return 'herb_benefits', 0.8
                
                return intent, confidence
                
        except Exception as e:
            logger.error(f"Error in ML intent classification: {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> tuple[str, float]:
        """
        Fallback classification using enhanced pattern matching.
        
        Args:
            query: User query text
            
        Returns:
            Tuple of (intent, confidence)
        """
        query_lower = query.lower()
        
        # Enhanced herb benefits patterns (more comprehensive)
        herb_patterns = [
            'benefits of', 'uses of', 'properties of', 'effects of',
            'what is', 'how does', 'medicinal properties', 'therapeutic effects',
            'advantages of', 'good for', 'help with', 'healing properties',
            'traditional uses', 'ayurvedic uses', 'health benefits'
        ]
        
        # Common herb names (comprehensive list)
        herb_names = [
            'turmeric', 'haridra', 'ginger', 'adrak', 'ashwagandha', 'tulsi', 'holy basil',
            'neem', 'nimba', 'brahmi', 'bacopa', 'triphala', 'amla', 'amalaki',
            'guduchi', 'giloy', 'shankhpushpi', 'arjuna', 'fenugreek', 'methi',
            'cinnamon', 'cardamom', 'cloves', 'black pepper', 'long pepper',
            'cumin', 'coriander', 'fennel', 'ajwain', 'hing', 'asafoetida'
        ]
        
        # Disease/symptom patterns
        disease_patterns = [
            'i have', 'suffering from', 'experiencing', 'symptoms of',
            'treatment for', 'cure for', 'medicine for', 'diagnosis',
            'pain', 'ache', 'fever', 'headache', 'migraine', 'cold', 'cough'
        ]
        
        # Check for herb benefits - prioritize this
        contains_herb = any(herb in query_lower for herb in herb_names)
        contains_herb_pattern = any(pattern in query_lower for pattern in herb_patterns)
        
        if contains_herb and contains_herb_pattern:
            return 'herb_benefits', 0.9
        elif contains_herb_pattern:
            # Even without specific herb name, if it has benefit patterns, likely herb query
            return 'herb_benefits', 0.7
        elif contains_herb:
            # Contains herb name but no clear benefit pattern
            return 'herb_benefits', 0.6
        
        # Check for disease/symptom queries
        for pattern in disease_patterns:
            if pattern in query_lower:
                return 'disease_prediction', 0.7
        
        # Default to general info
        return 'general_info', 0.5
    
    def get_intent_probabilities(self, query: str) -> dict[str, float]:
        """
        Get probabilities for all intent classes.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary mapping intent to probability
        """
        if not self.is_loaded or not self.model or not self.tokenizer:
            intent, confidence = self._fallback_classification(query)
            return {
                'herb_benefits': confidence if intent == 'herb_benefits' else 0.1,
                'disease_prediction': confidence if intent == 'disease_prediction' else 0.1,
                'general_info': confidence if intent == 'general_info' else 0.1
            }
        
        try:
            # Tokenize input
            encoding = self.tokenizer(
                query,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(encoding['input_ids'], encoding['attention_mask'])
                
                # Handle both dict and tensor outputs
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                # Get probabilities
                probabilities = torch.softmax(logits, dim=1).squeeze()
                
                # Map to labels
                result = {}
                for idx, prob in enumerate(probabilities):
                    label = self.label_mapping.get(str(idx), f'class_{idx}')
                    result[label] = prob.item()
                
                return result
                
        except Exception as e:
            logger.error(f"Error getting intent probabilities: {e}")
            intent, confidence = self._fallback_classification(query)
            return {
                'herb_benefits': confidence if intent == 'herb_benefits' else 0.1,
                'disease_prediction': confidence if intent == 'disease_prediction' else 0.1,
                'general_info': confidence if intent == 'general_info' else 0.1
            }
    
    def is_available(self) -> bool:
        """Check if the classifier is available."""
        return self.is_loaded

# Global service instance
_intent_classifier_service = None

def get_intent_classifier_service() -> QueryIntentClassifierService:
    """Get the global query intent classifier service instance."""
    global _intent_classifier_service
    if _intent_classifier_service is None:
        _intent_classifier_service = QueryIntentClassifierService()
    return _intent_classifier_service

# Example usage and testing
if __name__ == "__main__":
    service = QueryIntentClassifierService()
    
    test_queries = [
        "benefits of ginger",
        "what are the benefits of turmeric", 
        "I have a headache",
        "suffering from migraine",
        "what is ayurveda",
        "ginger uses",
        "turmeric properties",
        "experiencing fever"
    ]
    
    print("🧪 Testing Query Intent Classifier Service")
    print("=" * 50)
    
    for query in test_queries:
        intent, confidence = service.classify_intent(query)
        probabilities = service.get_intent_probabilities(query)
        
        print(f"Query: '{query}'")
        print(f"  Intent: {intent} (confidence: {confidence:.3f})")
        print(f"  Probabilities: {probabilities}")
        print("-" * 30)