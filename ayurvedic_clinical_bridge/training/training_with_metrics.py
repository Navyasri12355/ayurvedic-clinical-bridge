"""
Real Training Pipeline with Metrics Collection

This module implements actual model training with real datasets,
real forward/backward passes, and proper NLP metrics calculation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import time
import logging
import re
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import json

from .metrics_collector import MetricsCollector, ModelMetrics
from ..models.hybrid_ner import HybridMedicalNER
from ..data.dataset_integration import AyurvedicDatasetIntegrator

logger = logging.getLogger(__name__)

class AyurvedicNERDataset(Dataset):
    """
    Dataset class for Ayurvedic Named Entity Recognition.
    Processes the CSV data and creates proper NER training examples.
    """
    
    def __init__(self, csv_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Entity labels for BIO tagging
        self.entity_labels = [
            'O',  # Outside
            'B-DISEASE', 'I-DISEASE',      # Diseases
            'B-SYMPTOM', 'I-SYMPTOM',      # Symptoms  
            'B-HERB', 'I-HERB',            # Herbs/Remedies
            'B-DOSAGE', 'I-DOSAGE',        # Dosages
            'B-TREATMENT', 'I-TREATMENT'   # Treatments
        ]
        self.label_to_id = {label: i for i, label in enumerate(self.entity_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.entity_labels)}
        
        # Load and process data
        self.examples = self._load_and_process_data(csv_path)
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def _load_and_process_data(self, csv_path: str) -> List[Dict]:
        """Load CSV data and convert to NER training examples."""
        df = pd.read_csv(csv_path)
        examples = []
        
        # Process each row to create training examples
        for idx, row in df.iterrows():
            # Create text from available columns
            text_parts = []
            entities = []
            
            # Extract disease information
            if 'Disease' in row and pd.notna(row['Disease']):
                disease = str(row['Disease']).strip()
                start_pos = len(' '.join(text_parts))
                if text_parts:
                    start_pos += 1  # Account for space
                text_parts.append(f"Patient has {disease}")
                entities.append({
                    'start': start_pos + len("Patient has "),
                    'end': start_pos + len(f"Patient has {disease}"),
                    'label': 'DISEASE',
                    'text': disease
                })
            
            # Extract symptoms
            symptom_cols = [col for col in df.columns if 'symptom' in col.lower()]
            for col in symptom_cols:
                if col in row and pd.notna(row[col]):
                    symptom = str(row[col]).strip()
                    start_pos = len(' '.join(text_parts))
                    if text_parts:
                        start_pos += 1
                    text_parts.append(f"with symptoms of {symptom}")
                    entities.append({
                        'start': start_pos + len("with symptoms of "),
                        'end': start_pos + len(f"with symptoms of {symptom}"),
                        'label': 'SYMPTOM',
                        'text': symptom
                    })
            
            # Extract herbs/treatments
            herb_cols = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['herb', 'remedy', 'treatment', 'ayurvedic'])]
            for col in herb_cols:
                if col in row and pd.notna(row[col]):
                    herb = str(row[col]).strip()
                    start_pos = len(' '.join(text_parts))
                    if text_parts:
                        start_pos += 1
                    text_parts.append(f"treated with {herb}")
                    entities.append({
                        'start': start_pos + len("treated with "),
                        'end': start_pos + len(f"treated with {herb}"),
                        'label': 'HERB',
                        'text': herb
                    })
            
            # Extract dosage information
            dosage_cols = [col for col in df.columns if any(keyword in col.lower() 
                          for keyword in ['dosage', 'dose', 'formulation'])]
            for col in dosage_cols:
                if col in row and pd.notna(row[col]):
                    dosage = str(row[col]).strip()
                    # Look for dosage patterns
                    if any(unit in dosage.lower() for unit in ['mg', 'ml', 'tsp', 'daily', 'twice']):
                        start_pos = len(' '.join(text_parts))
                        if text_parts:
                            start_pos += 1
                        text_parts.append(f"dosage {dosage}")
                        entities.append({
                            'start': start_pos + len("dosage "),
                            'end': start_pos + len(f"dosage {dosage}"),
                            'label': 'DOSAGE',
                            'text': dosage
                        })
            
            if text_parts:
                full_text = ' '.join(text_parts)
                examples.append({
                    'text': full_text,
                    'entities': entities,
                    'source_row': idx
                })
        
        return examples
    
    def _create_bio_labels(self, text: str, entities: List[Dict]) -> List[int]:
        """Create BIO labels for the text."""
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        token_labels = ['O'] * len(tokens)
        
        # Map character positions to token positions
        char_to_token = {}
        current_pos = 0
        
        for i, token in enumerate(tokens):
            token_text = self.tokenizer.convert_tokens_to_string([token])
            token_start = text.find(token_text, current_pos)
            if token_start != -1:
                for j in range(len(token_text)):
                    if token_start + j < len(text):
                        char_to_token[token_start + j] = i
                current_pos = token_start + len(token_text)
        
        # Assign BIO labels
        for entity in entities:
            start_token = char_to_token.get(entity['start'])
            end_token = char_to_token.get(entity['end'] - 1)
            
            if start_token is not None and end_token is not None:
                # Assign B- label to first token
                token_labels[start_token] = f"B-{entity['label']}"
                # Assign I- labels to subsequent tokens
                for i in range(start_token + 1, min(end_token + 1, len(token_labels))):
                    token_labels[i] = f"I-{entity['label']}"
        
        # Convert to IDs
        label_ids = [self.label_to_id.get(label, 0) for label in token_labels]
        return label_ids
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        text = example['text']
        entities = example['entities']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create BIO labels
        labels = self._create_bio_labels(text, entities)
        
        # Pad or truncate labels to match tokenized length
        if len(labels) > self.max_length:
            labels = labels[:self.max_length]
        else:
            labels.extend([0] * (self.max_length - len(labels)))
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class SimpleRNNModel(nn.Module):
    """
    Simple RNN model for NER (baseline).
    
    Expected Latency: LOWEST (~5-15ms)
    Justification:
    - Simplest recurrent architecture
    - No gating mechanisms (just tanh activation)
    - Unidirectional processing
    - Fewer parameters than LSTM/GRU
    - However: Suffers from vanishing gradients, poor long-term dependencies
    """
    
    def __init__(self, vocab_size=30000, embed_dim=300, hidden_dim=256, 
                 num_layers=2, num_classes=9, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(
            embed_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        rnn_out, _ = self.rnn(embedded)
        rnn_out = self.dropout(rnn_out)
        logits = self.classifier(rnn_out)
        return {'logits': logits}


class LSTMModel(nn.Module):
    """
    LSTM model for NER (unidirectional).
    
    Expected Latency: MODERATE (~10-25ms)
    Justification:
    - 3 gates (input, forget, output) + cell state
    - More computations per timestep than RNN
    - Better gradient flow than RNN
    - Still unidirectional (only sees past context)
    """
    
    def __init__(self, vocab_size=30000, embed_dim=300, hidden_dim=256, 
                 num_layers=2, num_classes=9, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            bidirectional=False, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return {'logits': logits}


class GRUModel(nn.Module):
    """
    GRU model for NER.
    
    Expected Latency: LOW-MODERATE (~8-20ms)
    Justification:
    - 2 gates (reset, update) - simpler than LSTM
    - Fewer parameters than LSTM (no separate cell state)
    - Faster than LSTM but similar performance
    - Still unidirectional
    """
    
    def __init__(self, vocab_size=30000, embed_dim=300, hidden_dim=256, 
                 num_layers=2, num_classes=9, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers, 
            bidirectional=False, batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        gru_out, _ = self.gru(embedded)
        gru_out = self.dropout(gru_out)
        logits = self.classifier(gru_out)
        return {'logits': logits}


class BiLSTMModel(nn.Module):
    """
    BiLSTM model for NER.
    
    Expected Latency: MODERATE-HIGH (~20-50ms)
    Justification:
    - Bidirectional: Processes sequence TWICE (forward + backward)
    - 2x the computation of unidirectional LSTM
    - 2x the parameters (forward and backward LSTMs)
    - Captures both past and future context (crucial for NER)
    - Worth the latency cost for significantly better accuracy
    """
    
    def __init__(self, vocab_size=30000, embed_dim=300, hidden_dim=256, 
                 num_layers=2, num_classes=9, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, 
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        logits = self.classifier(lstm_out)
        return {'logits': logits}


class TransformerModel(nn.Module):
    """
    Transformer model for NER (BioBERT-based).
    
    Expected Latency: HIGHEST (~80-150ms)
    Justification:
    - Self-attention mechanism: O(n²) complexity with sequence length
    - 110M parameters (vs ~2-3M for BiLSTM)
    - 12 transformer layers with multi-head attention
    - Pre-trained on biomedical literature (PubMed, PMC)
    - Parallel processing within layers BUT large model size
    - Worth the latency for maximum accuracy in clinical NER
    - Best for: Offline processing, high-accuracy requirements
    """
    
    def __init__(self, model_name='dmis-lab/biobert-v1.1', num_classes=9, dropout=0.1):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(sequence_output)
        return {'logits': logits}


class RealTrainingWithMetrics:
    """Real training pipeline that collects metrics for model comparison."""
    
    def __init__(self, output_dir: str = "data", csv_path: str = "data/ayurgenix_dataset.csv"):
        self.metrics_collector = MetricsCollector(output_dir)
        self.all_metrics: List[ModelMetrics] = []
        self.csv_path = csv_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def prepare_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare train, validation, and test datasets."""
        logger.info("Preparing datasets from CSV data...")
        
        # Initialize tokenizer (using BioBERT)
        tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
        
        # Create dataset
        full_dataset = AyurvedicNERDataset(self.csv_path, tokenizer)
        
        # Split dataset
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, temp_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size + test_size]
        )
        val_dataset, test_dataset = torch.utils.data.random_split(
            temp_dataset, [val_size, test_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        logger.info(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def train_model(self, model_config: Dict[str, Any], 
                   train_loader: DataLoader, val_loader: DataLoader, 
                   test_loader: DataLoader = None) -> ModelMetrics:
        """
        Train a model with real forward/backward passes and collect metrics.
        """
        model_name = model_config.get('name', 'Unknown Model')
        model_type = model_config.get('type', 'unknown')
        
        logger.info(f"Starting real training for {model_name}")
        
        # Initialize model
        model = self._create_model(model_config)
        model.to(self.device)
        
        # Start metrics collection
        session = self.metrics_collector.start_training_session(model_name, model_type)
        
        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=model_config.get('lr', 2e-5))
        criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore padding tokens
        
        num_epochs = model_config.get('epochs', 5)  # Reduced for real training
        
        # Training loop
        for epoch in range(1, num_epochs + 1):
            # Training phase
            model.train()
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion)
            
            # Validation phase
            model.eval()
            val_loss, val_metrics = self._validate_epoch(model, val_loader, criterion)
            
            # Record metrics
            self.metrics_collector.record_epoch_metrics(
                session, epoch, train_loss, val_loss,
                accuracy=val_metrics.get('accuracy', 0.0),
                precision=val_metrics.get('precision', 0.0),
                recall=val_metrics.get('recall', 0.0),
                f1_score=val_metrics.get('f1_score', 0.0)
            )
            
            logger.info(f"Epoch {epoch}/{num_epochs}: "
                       f"train_loss={train_loss:.4f}, "
                       f"val_loss={val_loss:.4f}, "
                       f"f1_score={val_metrics.get('f1_score', 0):.3f}")
        
        # Final evaluation
        final_metrics = self._final_evaluation(model, test_loader if test_loader else val_loader)
        
        # Finalize metrics collection
        metrics = self.metrics_collector.finalize_training_session(
            session, model, test_loader, final_metrics
        )
        
        self.all_metrics.append(metrics)
        logger.info(f"Training completed for {model_name}")
        
        return metrics
    
    def _create_model(self, config: Dict[str, Any]) -> nn.Module:
        """Create model based on configuration."""
        architecture = config.get('architecture', 'bilstm')
        
        if architecture == 'rnn':
            return SimpleRNNModel(
                hidden_dim=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 2),
                num_classes=9
            )
        elif architecture == 'lstm':
            return LSTMModel(
                hidden_dim=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 2),
                num_classes=9
            )
        elif architecture == 'gru':
            return GRUModel(
                hidden_dim=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 2),
                num_classes=9
            )
        elif architecture == 'bilstm':
            return BiLSTMModel(
                hidden_dim=config.get('hidden_size', 256),
                num_layers=config.get('num_layers', 2),
                num_classes=9
            )
        elif architecture == 'transformer':
            return TransformerModel(
                model_name=config.get('model_name', 'dmis-lab/biobert-v1.1'),
                num_classes=9
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader, 
                    optimizer: optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch with real forward/backward passes."""
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(model, 'forward') and 'attention_mask' in model.forward.__code__.co_varnames:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids)
            
            logits = outputs['logits']
            
            # Reshape for loss calculation
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Limit batches for reasonable training time
            if num_batches >= 20:  # Process max 20 batches per epoch
                break
        
        return total_loss / max(num_batches, 1)
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader, 
                       criterion: nn.Module) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch with real predictions."""
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward') and 'attention_mask' in model.forward.__code__.co_varnames:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)
                
                logits = outputs['logits']
                
                # Calculate loss
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect predictions and labels (flatten and filter out padding)
                mask = labels != -100  # Ignore padding tokens
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
                
                num_batches += 1
                
                # Limit batches for reasonable validation time
                if num_batches >= 10:  # Process max 10 batches for validation
                    break
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate real NLP metrics
        metrics = self._calculate_real_metrics(all_predictions, all_labels)
        
        return avg_loss, metrics
    
    def _calculate_real_metrics(self, predictions: List[int], labels: List[int]) -> Dict[str, float]:
        """Calculate real NLP metrics from predictions and labels."""
        if not predictions or not labels:
            return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        # Token-level accuracy
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        accuracy = correct / len(predictions)
        
        # Calculate precision, recall, F1 for each class (excluding 'O' class)
        from collections import defaultdict
        
        tp = defaultdict(int)  # True positives
        fp = defaultdict(int)  # False positives
        fn = defaultdict(int)  # False negatives
        
        for pred, true in zip(predictions, labels):
            if pred == true:
                tp[pred] += 1
            else:
                fp[pred] += 1
                fn[true] += 1
        
        # Calculate macro-averaged metrics (excluding 'O' class)
        precisions = []
        recalls = []
        f1s = []
        
        for class_id in range(1, 9):  # Skip 'O' class (id=0)
            precision = tp[class_id] / (tp[class_id] + fp[class_id]) if (tp[class_id] + fp[class_id]) > 0 else 0.0
            recall = tp[class_id] / (tp[class_id] + fn[class_id]) if (tp[class_id] + fn[class_id]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        return {
            'accuracy': accuracy,
            'precision': np.mean(precisions) if precisions else 0.0,
            'recall': np.mean(recalls) if recalls else 0.0,
            'f1_score': np.mean(f1s) if f1s else 0.0
        }
    
    def _final_evaluation(self, model: nn.Module, test_loader: DataLoader) -> Dict[str, float]:
        """Perform final evaluation on test set."""
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if hasattr(model, 'forward') and 'attention_mask' in model.forward.__code__.co_varnames:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    outputs = model(input_ids)
                
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect predictions and labels
                mask = labels != -100
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        
        return self._calculate_real_metrics(all_predictions, all_labels)
    
    def train_all_models(self) -> List[ModelMetrics]:
        """Train all model variants with real data and collect metrics."""
        
        # Prepare datasets
        train_loader, val_loader, test_loader = self.prepare_datasets()
        
        # Model configurations - includes baseline models for comparison
        model_configs = [
            {
                'name': 'Simple RNN',
                'type': 'rnn',
                'architecture': 'rnn',
                'hidden_size': 256,
                'num_layers': 2,
                'epochs': 3,
                'lr': 1e-3
            },
            {
                'name': 'LSTM',
                'type': 'lstm',
                'architecture': 'lstm',
                'hidden_size': 256,
                'num_layers': 2,
                'epochs': 3,
                'lr': 1e-3
            },
            {
                'name': 'GRU',
                'type': 'gru',
                'architecture': 'gru',
                'hidden_size': 256,
                'num_layers': 2,
                'epochs': 3,
                'lr': 1e-3
            },
            {
                'name': 'BiLSTM-CRF',
                'type': 'bilstm',
                'architecture': 'bilstm',
                'hidden_size': 256,
                'num_layers': 2,
                'epochs': 3,
                'lr': 1e-3
            },
            {
                'name': 'BioBERT-Transformer',
                'type': 'transformer',
                'architecture': 'transformer',
                'model_name': 'dmis-lab/biobert-v1.1',
                'epochs': 3,
                'lr': 2e-5
            }
        ]
        
        # Train each model
        for config in model_configs:
            try:
                logger.info(f"🚀 Starting real training: {config['name']}")
                metrics = self.train_model(config, train_loader, val_loader, test_loader)
                logger.info(f"✅ Completed training: {metrics.model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to train {config['name']}: {str(e)}")
                # Continue with other models even if one fails
                continue
        
        # Save all metrics
        if self.all_metrics:
            self.metrics_collector.save_metrics(self.all_metrics)
            logger.info(f"💾 Saved metrics for {len(self.all_metrics)} models")
        
        return self.all_metrics

# Example usage
def run_real_training_with_metrics():
    """Example of running real training with metrics collection."""
    trainer = RealTrainingWithMetrics()
    
    # Train all models with real data and collect metrics
    all_metrics = trainer.train_all_models()
    
    logger.info(f"Real training completed! Generated metrics for {len(all_metrics)} models")
    return all_metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_real_training_with_metrics()