"""
NLP-Specific Metrics for Named Entity Recognition (NER) Models

This module implements proper NLP evaluation metrics for our Ayurvedic clinical NER models.
Explains how accuracy and F1-score are calculated for NLP tasks.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class NERMetrics:
    """Comprehensive NER evaluation metrics."""
    # Token-level metrics
    token_accuracy: float
    token_precision: float
    token_recall: float
    token_f1: float
    
    # Entity-level metrics (more important for NER)
    entity_precision: float
    entity_recall: float
    entity_f1: float
    
    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]]
    
    # Confusion matrix
    confusion_matrix: Optional[List[List[int]]] = None
    
    # Additional NER-specific metrics
    exact_match_ratio: float = 0.0
    partial_match_ratio: float = 0.0

class NLPMetricsCalculator:
    """
    Calculate NLP-specific metrics for Named Entity Recognition tasks.
    
    For our Ayurvedic clinical system, we have entity types like:
    - DISEASE (diabetes, hypertension)
    - SYMPTOM (fever, headache) 
    - HERB (turmeric, ashwagandha)
    - DOSAGE (500mg, twice daily)
    - TREATMENT (oil massage, meditation)
    """
    
    def __init__(self, entity_labels: List[str]):
        """
        Initialize with entity labels.
        
        Args:
            entity_labels: List of entity labels (e.g., ['O', 'B-DISEASE', 'I-DISEASE', 'B-HERB', 'I-HERB'])
        """
        self.entity_labels = entity_labels
        self.label_to_id = {label: i for i, label in enumerate(entity_labels)}
        self.id_to_label = {i: label for i, label in enumerate(entity_labels)}
    
    def calculate_ner_metrics(self, 
                            true_labels: List[List[int]], 
                            pred_labels: List[List[int]]) -> NERMetrics:
        """
        Calculate comprehensive NER metrics.
        
        Args:
            true_labels: Ground truth labels (batch_size, seq_len)
            pred_labels: Predicted labels (batch_size, seq_len)
            
        Returns:
            NERMetrics object with all calculated metrics
        """
        # Flatten labels for token-level metrics
        flat_true = [label for sequence in true_labels for label in sequence]
        flat_pred = [label for sequence in pred_labels for label in sequence]
        
        # Calculate token-level metrics
        token_metrics = self._calculate_token_level_metrics(flat_true, flat_pred)
        
        # Extract entities for entity-level metrics
        true_entities = self._extract_entities(true_labels)
        pred_entities = self._extract_entities(pred_labels)
        
        # Calculate entity-level metrics
        entity_metrics = self._calculate_entity_level_metrics(true_entities, pred_entities)
        
        # Calculate per-class metrics
        per_class_metrics = self._calculate_per_class_metrics(flat_true, flat_pred)
        
        # Calculate exact and partial match ratios
        exact_match = self._calculate_exact_match_ratio(true_entities, pred_entities)
        partial_match = self._calculate_partial_match_ratio(true_entities, pred_entities)
        
        return NERMetrics(
            token_accuracy=token_metrics['accuracy'],
            token_precision=token_metrics['precision'],
            token_recall=token_metrics['recall'],
            token_f1=token_metrics['f1'],
            entity_precision=entity_metrics['precision'],
            entity_recall=entity_metrics['recall'],
            entity_f1=entity_metrics['f1'],
            per_class_metrics=per_class_metrics,
            exact_match_ratio=exact_match,
            partial_match_ratio=partial_match
        )
    
    def _calculate_token_level_metrics(self, true_labels: List[int], pred_labels: List[int]) -> Dict[str, float]:
        """
        Calculate token-level accuracy, precision, recall, F1.
        
        TOKEN-LEVEL ACCURACY CALCULATION:
        - For each token in the sequence, check if predicted label == true label
        - Accuracy = (Number of correctly predicted tokens) / (Total tokens)
        
        Example:
        True:  [O, B-DISEASE, I-DISEASE, O, B-HERB]
        Pred:  [O, B-DISEASE, O,        O, B-HERB]
        Correct: ✓     ✓        ✗       ✓    ✓
        Accuracy = 4/5 = 0.8 (80%)
        """
        correct = sum(1 for t, p in zip(true_labels, pred_labels) if t == p)
        total = len(true_labels)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate precision, recall, F1 for each class
        tp = defaultdict(int)  # True positives
        fp = defaultdict(int)  # False positives  
        fn = defaultdict(int)  # False negatives
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            if true_label == pred_label:
                tp[true_label] += 1
            else:
                fp[pred_label] += 1
                fn[true_label] += 1
        
        # Macro-averaged metrics (average across all classes)
        precisions = []
        recalls = []
        f1s = []
        
        for label_id in range(len(self.entity_labels)):
            precision = tp[label_id] / (tp[label_id] + fp[label_id]) if (tp[label_id] + fp[label_id]) > 0 else 0.0
            recall = tp[label_id] / (tp[label_id] + fn[label_id]) if (tp[label_id] + fn[label_id]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        return {
            'accuracy': accuracy,
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s)
        }
    
    def _extract_entities(self, label_sequences: List[List[int]]) -> List[List[Tuple[str, int, int]]]:
        """
        Extract entities from BIO-tagged sequences.
        
        ENTITY EXTRACTION EXPLANATION:
        - BIO tagging: B-TYPE (beginning), I-TYPE (inside), O (outside)
        - Entity = continuous sequence of B-TYPE followed by I-TYPE tags
        
        Example:
        Tokens: ["Patient", "has", "diabetes", "mellitus", "and", "fever"]
        Labels: [O, O, B-DISEASE, I-DISEASE, O, B-SYMPTOM]
        Entities: [("DISEASE", 2, 4), ("SYMPTOM", 5, 6)]
        """
        all_entities = []
        
        for sequence_labels in label_sequences:
            entities = []
            current_entity = None
            
            for i, label_id in enumerate(sequence_labels):
                label = self.id_to_label[label_id]
                
                if label.startswith('B-'):
                    # Start of new entity
                    if current_entity:
                        entities.append(current_entity)
                    entity_type = label[2:]  # Remove 'B-' prefix
                    current_entity = (entity_type, i, i + 1)
                
                elif label.startswith('I-') and current_entity:
                    # Continue current entity
                    entity_type = label[2:]  # Remove 'I-' prefix
                    if entity_type == current_entity[0]:
                        current_entity = (current_entity[0], current_entity[1], i + 1)
                    else:
                        # Entity type mismatch, end current entity
                        entities.append(current_entity)
                        current_entity = None
                
                else:
                    # 'O' tag or entity type mismatch
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
            
            # Add final entity if exists
            if current_entity:
                entities.append(current_entity)
            
            all_entities.append(entities)
        
        return all_entities
    
    def _calculate_entity_level_metrics(self, true_entities: List[List[Tuple]], pred_entities: List[List[Tuple]]) -> Dict[str, float]:
        """
        Calculate entity-level precision, recall, F1.
        
        ENTITY-LEVEL F1 CALCULATION:
        - Precision = (Correctly predicted entities) / (Total predicted entities)
        - Recall = (Correctly predicted entities) / (Total true entities)
        - F1 = 2 * (Precision * Recall) / (Precision + Recall)
        
        Example:
        True entities: [("DISEASE", 2, 4), ("SYMPTOM", 5, 6)]
        Pred entities: [("DISEASE", 2, 4), ("HERB", 7, 8)]
        
        Correct predictions: 1 (the DISEASE entity)
        Precision = 1/2 = 0.5 (1 correct out of 2 predicted)
        Recall = 1/2 = 0.5 (1 correct out of 2 true entities)
        F1 = 2 * 0.5 * 0.5 / (0.5 + 0.5) = 0.5
        """
        total_true = 0
        total_pred = 0
        total_correct = 0
        
        for true_seq, pred_seq in zip(true_entities, pred_entities):
            true_set = set(true_seq)
            pred_set = set(pred_seq)
            
            total_true += len(true_set)
            total_pred += len(pred_set)
            total_correct += len(true_set.intersection(pred_set))
        
        precision = total_correct / total_pred if total_pred > 0 else 0.0
        recall = total_correct / total_true if total_true > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_per_class_metrics(self, true_labels: List[int], pred_labels: List[int]) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall, F1 for each entity class."""
        per_class = {}
        
        for label_id, label_name in self.id_to_label.items():
            if label_name == 'O':  # Skip 'Outside' tag
                continue
                
            # Binary classification for this class
            true_binary = [1 if label == label_id else 0 for label in true_labels]
            pred_binary = [1 if label == label_id else 0 for label in pred_labels]
            
            tp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class[label_name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': sum(true_binary)
            }
        
        return per_class
    
    def _calculate_exact_match_ratio(self, true_entities: List[List[Tuple]], pred_entities: List[List[Tuple]]) -> float:
        """Calculate ratio of sequences with exact entity matches."""
        exact_matches = 0
        total_sequences = len(true_entities)
        
        for true_seq, pred_seq in zip(true_entities, pred_entities):
            if set(true_seq) == set(pred_seq):
                exact_matches += 1
        
        return exact_matches / total_sequences if total_sequences > 0 else 0.0
    
    def _calculate_partial_match_ratio(self, true_entities: List[List[Tuple]], pred_entities: List[List[Tuple]]) -> float:
        """Calculate ratio of entities with partial overlap."""
        partial_matches = 0
        total_entities = 0
        
        for true_seq, pred_seq in zip(true_entities, pred_entities):
            total_entities += len(true_seq)
            
            for true_entity in true_seq:
                for pred_entity in pred_seq:
                    # Check for partial overlap
                    if (true_entity[0] == pred_entity[0] and  # Same entity type
                        not (true_entity[2] <= pred_entity[1] or pred_entity[2] <= true_entity[1])):  # Overlapping spans
                        partial_matches += 1
                        break
        
        return partial_matches / total_entities if total_entities > 0 else 0.0

def demonstrate_nlp_metrics():
    """Demonstrate how NLP metrics are calculated with examples."""
    print("=" * 80)
    print("NLP METRICS EXPLANATION FOR AYURVEDIC CLINICAL NER")
    print("=" * 80)
    
    print("\n1. TOKEN-LEVEL ACCURACY:")
    print("   - Measures percentage of tokens correctly classified")
    print("   - Example: 'Patient has diabetes' -> [O, O, B-DISEASE]")
    print("   - If predicted [O, O, B-DISEASE] -> 100% accuracy")
    print("   - If predicted [O, B-SYMPTOM, B-DISEASE] -> 66.7% accuracy")
    
    print("\n2. ENTITY-LEVEL F1-SCORE:")
    print("   - More important for NER tasks")
    print("   - Precision = Correct entities / Predicted entities")
    print("   - Recall = Correct entities / True entities")
    print("   - F1 = Harmonic mean of precision and recall")
    
    print("\n3. WHY THESE METRICS ARE VALID FOR NLP:")
    print("   ✓ Token accuracy shows model's ability to classify individual words")
    print("   ✓ Entity F1 shows model's ability to extract complete medical entities")
    print("   ✓ Per-class metrics show performance on specific entity types")
    print("   ✓ Standard metrics used in medical NER research")
    
    print("\n4. AYURVEDIC CLINICAL EXAMPLE:")
    print("   Text: 'Patient has diabetes, prescribe turmeric 500mg'")
    print("   True entities: [DISEASE: diabetes], [HERB: turmeric], [DOSAGE: 500mg]")
    print("   Pred entities: [DISEASE: diabetes], [HERB: turmeric]")
    print("   Entity Precision: 2/2 = 100% (both predicted entities correct)")
    print("   Entity Recall: 2/3 = 66.7% (missed dosage entity)")
    print("   Entity F1: 2 * 1.0 * 0.667 / (1.0 + 0.667) = 80%")
    
    # Create example calculator
    labels = ['O', 'B-DISEASE', 'I-DISEASE', 'B-HERB', 'I-HERB', 'B-DOSAGE', 'I-DOSAGE']
    calculator = NLPMetricsCalculator(labels)
    
    # Example sequences
    true_labels = [[0, 0, 1, 2, 0, 3, 0, 5]]  # "O O B-DISEASE I-DISEASE O B-HERB O B-DOSAGE"
    pred_labels = [[0, 0, 1, 2, 0, 3, 0, 0]]   # "O O B-DISEASE I-DISEASE O B-HERB O O" (missed dosage)
    
    metrics = calculator.calculate_ner_metrics(true_labels, pred_labels)
    
    print(f"\n5. CALCULATED METRICS:")
    print(f"   Token Accuracy: {metrics.token_accuracy:.3f}")
    print(f"   Token F1: {metrics.token_f1:.3f}")
    print(f"   Entity Precision: {metrics.entity_precision:.3f}")
    print(f"   Entity Recall: {metrics.entity_recall:.3f}")
    print(f"   Entity F1: {metrics.entity_f1:.3f}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    demonstrate_nlp_metrics()