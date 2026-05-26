"""
Training script for BiLSTM-CRF model

This script trains a BiLSTM-CRF model for clinical entity recognition
using the same data as the BioBERT model but optimized for speed.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Any
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import time
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src" / "backend"))

from ayurvedic_clinical_bridge.models.bilstm_crf_model import (
    BiLSTMCRF, BiLSTMCRFConfig, BiLSTMCRFTokenizer, create_default_vocab
)
from ayurvedic_clinical_bridge.models.biobert_transformer import ClinicalEntityType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalNERDataset(Dataset):
    """Dataset for clinical NER training."""
    
    def __init__(self, texts: List[str], labels: List[List[int]], 
                 tokenizer: BiLSTMCRFTokenizer, max_length: int = 256):
        """
        Initialize dataset.
        
        Args:
            texts: List of input texts
            labels: List of label sequences (same length as texts)
            tokenizer: BiLSTM-CRF tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        text = self.texts[idx]
        labels = self.labels[idx]
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Add special tokens
        tokens = [self.tokenizer.cls_token] + tokens[:self.max_length-2] + [self.tokenizer.sep_token]
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Prepare labels - align with tokenized input
        # For simplicity, we'll use the first len(tokens) labels and pad/truncate as needed
        aligned_labels = [0]  # CLS token gets O label
        
        # Add labels for actual tokens (skip CLS)
        for i in range(1, len(tokens) - 1):  # Skip CLS and SEP
            if i - 1 < len(labels):
                aligned_labels.append(labels[i - 1])
            else:
                aligned_labels.append(0)  # O label for padding
        
        aligned_labels.append(0)  # SEP token gets O label
        
        # Pad sequences
        while len(input_ids) < self.max_length:
            input_ids.append(self.tokenizer.pad_token_id)
            attention_mask.append(0)
            aligned_labels.append(-100)  # Ignore padding in loss
        
        # Truncate if necessary
        input_ids = input_ids[:self.max_length]
        attention_mask = attention_mask[:self.max_length]
        aligned_labels = aligned_labels[:self.max_length]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }


def create_synthetic_ner_data() -> Tuple[List[str], List[List[int]]]:
    """
    Create synthetic NER training data from available datasets.
    
    Returns:
        Tuple of (texts, label_sequences)
    """
    logger.info("Creating synthetic NER training data...")
    
    # Entity patterns with BIO tagging
    entity_patterns = {
        'HERB': [
            'turmeric', 'ginger', 'ashwagandha', 'brahmi', 'gudmar', 'arjuna',
            'tulsi', 'neem', 'amla', 'triphala', 'guggulu', 'shatavari',
            'licorice', 'fenugreek', 'cinnamon', 'cardamom', 'cumin',
            'gymnema sylvestre', 'terminalia arjuna', 'withania somnifera'
        ],
        'DISEASE': [
            'diabetes', 'hypertension', 'arthritis', 'asthma', 'migraine',
            'anxiety', 'depression', 'insomnia', 'obesity', 'fever',
            'cough', 'cold', 'indigestion', 'constipation', 'acidity',
            'heart disease', 'kidney disease', 'liver disease'
        ],
        'SYMPTOM': [
            'pain', 'inflammation', 'swelling', 'headache', 'nausea',
            'fatigue', 'dizziness', 'burning', 'stiffness', 'weakness',
            'thirst', 'frequent urination', 'blurred vision'
        ],
        'DOSAGE': [
            '500mg', '250mg', '1000mg', 'twice daily', 'three times',
            'daily', 'mg', 'grams', 'tsp', 'capsule', 'tablet'
        ]
    }
    
    # Create label mappings
    label_to_id = {'O': 0}  # Outside
    for entity_type in entity_patterns.keys():
        label_to_id[f'B_{entity_type}'] = len(label_to_id)
        label_to_id[f'I_{entity_type}'] = len(label_to_id)
    
    # Template sentences for synthetic data
    templates = [
        "I have {DISEASE} and need treatment with {HERB}.",
        "Can {HERB} help with {SYMPTOM}?",
        "Take {DOSAGE} of {HERB} for {DISEASE}.",
        "Patient shows {SYMPTOM} due to {DISEASE}.",
        "{HERB} is effective for treating {DISEASE} and {SYMPTOM}.",
        "Recommended dosage is {DOSAGE} of {HERB} daily.",
        "Chronic {DISEASE} causes severe {SYMPTOM}.",
        "Natural remedy {HERB} reduces {SYMPTOM} in {DISEASE}.",
        "Clinical trial shows {HERB} at {DOSAGE} improves {DISEASE}.",
        "Ayurvedic treatment uses {HERB} for {DISEASE} management."
    ]
    
    texts = []
    label_sequences = []
    
    import random
    random.seed(42)
    
    # Generate synthetic examples
    for _ in range(1000):  # Generate 1000 examples
        template = random.choice(templates)
        text = template
        entities_in_text = []
        
        # Replace placeholders with actual entities
        for entity_type, entity_list in entity_patterns.items():
            placeholder = f"{{{entity_type}}}"
            if placeholder in text:
                entity = random.choice(entity_list)
                start_pos = text.find(placeholder)
                text = text.replace(placeholder, entity, 1)
                entities_in_text.append({
                    'type': entity_type,
                    'text': entity,
                    'start': start_pos,
                    'end': start_pos + len(entity)
                })
        
        # Create BIO labels
        words = text.lower().split()
        labels = [label_to_id['O']] * len(words)
        
        # Assign entity labels
        for entity in entities_in_text:
            entity_words = entity['text'].lower().split()
            entity_start_word = None
            
            # Find entity position in word sequence
            for i in range(len(words) - len(entity_words) + 1):
                if words[i:i+len(entity_words)] == entity_words:
                    entity_start_word = i
                    break
            
            if entity_start_word is not None:
                # Assign B- and I- labels
                labels[entity_start_word] = label_to_id[f'B_{entity["type"]}']
                for j in range(1, len(entity_words)):
                    if entity_start_word + j < len(labels):
                        labels[entity_start_word + j] = label_to_id[f'I_{entity["type"]}']
        
        texts.append(text)
        label_sequences.append(labels)
    
    # Add some real examples from available data
    try:
        qa_data_path = Path("data/datasets/ayurvedic_qa_processed.csv")
        if qa_data_path.exists():
            df = pd.read_csv(qa_data_path)
            
            # Process Q&A data for entity extraction
            for _, row in df.head(200).iterrows():  # Use first 200 rows
                question = str(row.get('Question', ''))
                answer = str(row.get('Answer', ''))
                
                for text_sample in [question, answer]:
                    if len(text_sample) > 10:  # Skip very short texts
                        # Simple entity labeling for real data
                        words = text_sample.lower().split()
                        labels = [label_to_id['O']] * len(words)
                        
                        # Find entities in text
                        for entity_type, entity_list in entity_patterns.items():
                            for entity in entity_list:
                                entity_words = entity.split()
                                for i in range(len(words) - len(entity_words) + 1):
                                    if words[i:i+len(entity_words)] == entity_words:
                                        labels[i] = label_to_id[f'B_{entity_type}']
                                        for j in range(1, len(entity_words)):
                                            if i + j < len(labels):
                                                labels[i + j] = label_to_id[f'I_{entity_type}']
                        
                        texts.append(text_sample)
                        label_sequences.append(labels)
    
    except Exception as e:
        logger.warning(f"Could not load real data: {e}")
    
    logger.info(f"Created {len(texts)} training examples with {len(label_to_id)} labels")
    
    # Save label mappings
    label_mapping_path = Path("models/bilstm_crf/label_mappings.json")
    label_mapping_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(label_mapping_path, 'w') as f:
        json.dump(label_to_id, f, indent=2)
    
    return texts, label_sequences


def train_bilstm_crf(
    texts: List[str],
    label_sequences: List[List[int]],
    config: BiLSTMCRFConfig,
    vocab_to_idx: Dict[str, int],
    save_dir: str = "models/bilstm_crf",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    validation_split: float = 0.2
) -> BiLSTMCRF:
    """
    Train BiLSTM-CRF model.
    
    Args:
        texts: Training texts
        label_sequences: Corresponding label sequences
        config: Model configuration
        vocab_to_idx: Vocabulary mapping
        save_dir: Directory to save model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        validation_split: Fraction of data for validation
        
    Returns:
        Trained model
    """
    logger.info("Starting BiLSTM-CRF training...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = BiLSTMCRFTokenizer(vocab_to_idx)
    model = BiLSTMCRF(config, vocab_to_idx)
    model.to(device)
    
    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, label_sequences, test_size=validation_split, random_state=42
    )
    
    # Create datasets
    train_dataset = ClinicalNERDataset(train_texts, train_labels, tokenizer, config.max_length)
    val_dataset = ClinicalNERDataset(val_texts, val_labels, tokenizer, config.max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_true_labels = []
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Collect predictions for metrics
            if 'predictions' in outputs:
                predictions = outputs['predictions'].cpu().numpy()
                true_labels = batch['labels'].cpu().numpy()
                
                # Flatten and filter out padding
                for pred_seq, true_seq in zip(predictions, true_labels):
                    valid_indices = true_seq != -100
                    train_predictions.extend(pred_seq[valid_indices])
                    train_true_labels.extend(true_seq[valid_indices])
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs['loss']
                total_val_loss += loss.item()
                
                # Collect predictions
                if 'predictions' in outputs:
                    predictions = outputs['predictions'].cpu().numpy()
                    true_labels = batch['labels'].cpu().numpy()
                    
                    for pred_seq, true_seq in zip(predictions, true_labels):
                        valid_indices = true_seq != -100
                        val_predictions.extend(pred_seq[valid_indices])
                        val_true_labels.extend(true_seq[valid_indices])
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Calculate metrics
        train_f1 = f1_score(train_true_labels, train_predictions, average='weighted', zero_division=0)
        val_f1 = f1_score(val_true_labels, val_predictions, average='weighted', zero_division=0)
        
        logger.info(f"Epoch {epoch+1}/{epochs}:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train F1: {train_f1:.4f}")
        logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val F1: {val_f1:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            logger.info(f"  New best model saved (Val Loss: {best_val_loss:.4f})")
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save model
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'vocab_to_idx': vocab_to_idx,
        'best_val_loss': best_val_loss
    }
    
    torch.save(checkpoint, save_path / "pytorch_model.bin")
    
    # Save configuration
    with open(save_path / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Save tokenizer
    tokenizer.save_pretrained(str(save_path))
    
    logger.info(f"Model saved to {save_path}")
    
    return model


def evaluate_model(model: BiLSTMCRF, tokenizer: BiLSTMCRFTokenizer, 
                  test_texts: List[str], test_labels: List[List[int]],
                  device: torch.device) -> Dict[str, float]:
    """Evaluate trained model on test data."""
    logger.info("Evaluating model...")
    
    model.eval()
    test_dataset = ClinicalNERDataset(test_texts, test_labels, tokenizer, model.config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            predictions = outputs['predictions'].cpu().numpy()
            true_labels = batch['labels'].cpu().numpy()
            
            for pred_seq, true_seq in zip(predictions, true_labels):
                valid_indices = true_seq != -100
                all_predictions.extend(pred_seq[valid_indices])
                all_true_labels.extend(true_seq[valid_indices])
    
    # Calculate metrics
    f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    
    # Detailed classification report
    try:
        # Load label mappings
        label_mapping_path = Path("models/bilstm_crf/label_mappings.json")
        if label_mapping_path.exists():
            with open(label_mapping_path, 'r') as f:
                label_to_id = json.load(f)
            id_to_label = {v: k for k, v in label_to_id.items()}
            
            target_names = [id_to_label.get(i, f'LABEL_{i}') for i in range(max(all_true_labels) + 1)]
            report = classification_report(all_true_labels, all_predictions, 
                                         target_names=target_names, zero_division=0)
            logger.info(f"Classification Report:\n{report}")
    
    except Exception as e:
        logger.warning(f"Could not generate detailed report: {e}")
    
    metrics = {
        'f1_score': f1,
        'accuracy': sum(p == t for p, t in zip(all_predictions, all_true_labels)) / len(all_predictions)
    }
    
    logger.info(f"Test F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    
    return metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BiLSTM-CRF for clinical NER')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--embedding_dim', type=int, default=300, help='Embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--save_dir', type=str, default='models/bilstm_crf', help='Save directory')
    parser.add_argument('--use_synthetic', action='store_true', help='Use synthetic data')
    
    args = parser.parse_args()
    
    logger.info("Starting BiLSTM-CRF training script...")
    
    try:
        # Create synthetic training data
        texts, label_sequences = create_synthetic_ner_data()
        
        if not texts:
            logger.error("No training data available")
            return
        
        # Create vocabulary from training data
        vocab_to_idx = create_default_vocab()
        
        # Add words from training texts to vocabulary
        for text in texts:
            words = text.lower().split()
            for word in words:
                if word not in vocab_to_idx:
                    vocab_to_idx[word] = len(vocab_to_idx)
        
        # Create configuration
        config = BiLSTMCRFConfig(
            vocab_size=len(vocab_to_idx),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size
        )
        
        logger.info(f"Configuration: {config}")
        
        # Train model
        model = train_bilstm_crf(
            texts=texts,
            label_sequences=label_sequences,
            config=config,
            vocab_to_idx=vocab_to_idx,
            save_dir=args.save_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Evaluate on test set (use last 20% of data)
        split_idx = int(0.8 * len(texts))
        test_texts = texts[split_idx:]
        test_labels = label_sequences[split_idx:]
        
        if test_texts:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = BiLSTMCRFTokenizer(vocab_to_idx)
            metrics = evaluate_model(model, tokenizer, test_texts, test_labels, device)
            
            # Save metrics
            metrics_path = Path(args.save_dir) / "training_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
    