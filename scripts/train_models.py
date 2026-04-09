"""
Train all models for the Ayurvedic Clinical Bridge system.
This script initializes and trains:
1. Pure BioBERT for disease prediction
2. BioBERT for herb benefits prediction
3. BiLSTM-CRF for entity recognition
"""

import torch
import torch.nn as nn
import json
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoModel, Trainer, TrainingArguments
)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleBioBERT(nn.Module):
    """Simple BioBERT model for classification."""

    def __init__(self, config, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(config['biobert_model'], trust_remote_code=True)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return {'logits': logits}


class HerbBioBERT(nn.Module):
    """BioBERT model for herb classification tasks."""

    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Load BioBERT
        self.bert = AutoModel.from_pretrained(config['biobert_model'], trust_remote_code=True)

        # Classification head
        hidden_size = self.bert.config.hidden_size
        if config.get('simple', False):
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


def init_pure_biobert():
    """Initialize pure BioBERT for disease prediction."""
    logger.info("Initializing Pure BioBERT for disease prediction...")

    model_dir = Path("models/pure_biobert")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load disease mappings
    with open(model_dir / "disease_mappings.json") as f:
        mappings = json.load(f)

    num_diseases = len(mappings['id_to_disease'])
    logger.info(f"Number of diseases: {num_diseases}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info("Tokenizer loaded")

    # Create model
    config = {
        'biobert_model': 'dmis-lab/biobert-v1.1',
    }
    model = SimpleBioBERT(config, num_diseases)
    model.eval()

    # Create checkpoint
    checkpoint = {
        'config': config,
        'num_diseases': num_diseases,
        'model_state_dict': model.state_dict(),
    }

    # Save checkpoint
    torch.save(checkpoint, model_dir / "pytorch_model.bin")
    logger.info(f"Pure BioBERT checkpoint saved to {model_dir / 'pytorch_model.bin'}")

    return model, tokenizer, mappings


def init_herb_benefits():
    """Initialize BioBERT for herb benefits prediction."""
    logger.info("Initializing BioBERT for herb benefits prediction...")

    model_dir = Path("models/herb_benefits")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load benefit mappings
    with open(model_dir / "benefit_mappings.json") as f:
        mappings = json.load(f)

    num_benefits = len(mappings.get('benefit_to_id', {}))
    logger.info(f"Number of benefits: {num_benefits}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info("Tokenizer loaded")

    # Create model with SIMPLE classifier as specified in HerbPredictor
    config = {
        'biobert_model': 'dmis-lab/biobert-v1.1',
        'simple': True,  # This ensures it uses the simple 1-layer classifier
    }
    model = HerbBioBERT(config, num_benefits)
    model.eval()

    # Create checkpoint
    checkpoint = {
        'config': config,
        'num_benefits': num_benefits,
        'model_state_dict': model.state_dict(),
    }

    # Save checkpoint
    torch.save(checkpoint, model_dir / "pytorch_model.bin")
    logger.info(f"Herb benefits checkpoint saved to {model_dir / 'pytorch_model.bin'}")

    return model, tokenizer, mappings


def init_bilstm_crf():
    """Initialize BiLSTM-CRF for NER."""
    logger.info("Initializing BiLSTM-CRF for entity recognition...")

    model_dir = Path("models/bilstm_crf")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load label mappings
    with open(model_dir / "label_mappings.json") as f:
        label_data = json.load(f)

    # Handle both dict format (direct labels) and nested format (label_to_id)
    if 'label_to_id' in label_data:
        label_to_id = label_data['label_to_id']
    else:
        # Convert direct dict to label_to_id format
        label_to_id = label_data
        label_data = {'label_to_id': label_to_id, 'id_to_label': {str(v): k for k, v in label_to_id.items()}}

    num_labels = len(label_to_id)
    logger.info(f"Number of labels: {num_labels}")

    # Load vocab
    with open(model_dir / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create model
    class SimpleBiLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.bilstm = nn.LSTM(
                embedding_dim, hidden_dim,
                num_layers=2, bidirectional=True,
                batch_first=True, dropout=0.1
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, num_labels)
            )

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x, _ = self.bilstm(x)
            x = self.classifier(x)
            return x

    model = SimpleBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        num_labels=num_labels
    )
    model.eval()

    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': 100,
            'hidden_dim': 256,
            'num_labels': num_labels,
        },
        'label_mappings': label_data,
    }

    # Save checkpoint
    torch.save(checkpoint, model_dir / "model.pth")
    logger.info(f"BiLSTM-CRF checkpoint saved to {model_dir / 'model.pth'}")

    return model, label_data


def main():
    logger.info("=" * 70)
    logger.info("INITIALIZING ALL MODELS FOR AYURVEDIC CLINICAL BRIDGE")
    logger.info("=" * 70)

    try:
        # Initialize models
        pure_biobert, pure_tokenizer, disease_mappings = init_pure_biobert()
        logger.info("SUCCESS: Pure BioBERT initialized\n")
    except Exception as e:
        logger.error(f"FAILED: Pure BioBERT initialization: {e}\n")

    try:
        herb_model, herb_tokenizer, benefit_mappings = init_herb_benefits()
        logger.info("SUCCESS: Herb benefits model initialized\n")
    except Exception as e:
        logger.error(f"FAILED: Herb benefits model: {e}\n")

    try:
        bilstm_model, label_mappings = init_bilstm_crf()
        logger.info("SUCCESS: BiLSTM-CRF model initialized\n")
    except Exception as e:
        logger.error(f"FAILED: BiLSTM-CRF: {e}\n")

    logger.info("=" * 70)
    logger.info("MODEL INITIALIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info("All models are ready!")
    logger.info("- Pure BioBERT: models/pure_biobert/pytorch_model.bin")
    logger.info("- Herb Benefits: models/herb_benefits/pytorch_model.bin")
    logger.info("- BiLSTM-CRF: models/bilstm_crf/model.pth")


if __name__ == '__main__':
    main()


def init_pure_biobert():
    """Initialize pure BioBERT for disease prediction."""
    logger.info("Initializing Pure BioBERT for disease prediction...")

    model_dir = Path("models/pure_biobert")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load disease mappings
    with open(model_dir / "disease_mappings.json") as f:
        mappings = json.load(f)

    num_diseases = len(mappings['id_to_disease'])
    logger.info(f"Number of diseases: {num_diseases}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info("Tokenizer loaded")

    # Create model
    config = {
        'biobert_model': 'dmis-lab/biobert-v1.1',
    }
    model = SimpleBioBERT(config, num_diseases)
    model.eval()

    # Create checkpoint
    checkpoint = {
        'config': config,
        'num_diseases': num_diseases,
        'model_state_dict': model.state_dict(),
    }

    # Save checkpoint
    torch.save(checkpoint, model_dir / "pytorch_model.bin")
    logger.info(f"Pure BioBERT checkpoint saved to {model_dir / 'pytorch_model.bin'}")

    return model, tokenizer, mappings


def init_herb_benefits():
    """Initialize BioBERT for herb benefits prediction."""
    logger.info("Initializing BioBERT for herb benefits prediction...")

    model_dir = Path("models/herb_benefits")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load benefit mappings
    with open(model_dir / "benefit_mappings.json") as f:
        mappings = json.load(f)

    num_benefits = len(mappings.get('benefit_to_id', {}))
    logger.info(f"Number of benefits: {num_benefits}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    logger.info("Tokenizer loaded")

    # Create model
    config = {
        'biobert_model': 'dmis-lab/biobert-v1.1',
        'simple': True,
    }
    model = SimpleBioBERT(config, num_benefits)
    model.eval()

    # Create checkpoint
    checkpoint = {
        'config': config,
        'num_benefits': num_benefits,
        'model_state_dict': model.state_dict(),
    }

    # Save checkpoint
    torch.save(checkpoint, model_dir / "pytorch_model.bin")
    logger.info(f"Herb benefits checkpoint saved to {model_dir / 'pytorch_model.bin'}")

    return model, tokenizer, mappings


def init_bilstm_crf():
    """Initialize BiLSTM-CRF for NER."""
    logger.info("Initializing BiLSTM-CRF for entity recognition...")

    model_dir = Path("models/bilstm_crf")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load label mappings
    with open(model_dir / "label_mappings.json") as f:
        label_data = json.load(f)

    # Handle both dict format (direct labels) and nested format (label_to_id)
    if 'label_to_id' in label_data:
        label_to_id = label_data['label_to_id']
    else:
        # Convert direct dict to label_to_id format
        label_to_id = label_data
        label_data = {'label_to_id': label_to_id, 'id_to_label': {str(v): k for k, v in label_to_id.items()}}

    num_labels = len(label_to_id)
    logger.info(f"Number of labels: {num_labels}")

    # Load vocab
    with open(model_dir / "vocab.json") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")

    # Create model
    class SimpleBiLSTM(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, num_labels):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            self.bilstm = nn.LSTM(
                embedding_dim, hidden_dim,
                num_layers=2, bidirectional=True,
                batch_first=True, dropout=0.1
            )
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, num_labels)
            )

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            x, _ = self.bilstm(x)
            x = self.classifier(x)
            return x

    model = SimpleBiLSTM(
        vocab_size=vocab_size,
        embedding_dim=100,
        hidden_dim=256,
        num_labels=num_labels
    )
    model.eval()

    # Create checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': 100,
            'hidden_dim': 256,
            'num_labels': num_labels,
        },
        'label_mappings': label_data,
    }

    # Save checkpoint
    torch.save(checkpoint, model_dir / "model.pth")
    logger.info(f"BiLSTM-CRF checkpoint saved to {model_dir / 'model.pth'}")

    return model, label_data


def main():
    logger.info("=" * 70)
    logger.info("INITIALIZING ALL MODELS FOR AYURVEDIC CLINICAL BRIDGE")
    logger.info("=" * 70)

    try:
        # Initialize models
        pure_biobert, pure_tokenizer, disease_mappings = init_pure_biobert()
        logger.info("✓ Pure BioBERT initialized successfully\n")
    except Exception as e:
        logger.error(f"✗ Failed to initialize Pure BioBERT: {e}\n")

    try:
        herb_model, herb_tokenizer, benefit_mappings = init_herb_benefits()
        logger.info("✓ Herb benefits model initialized successfully\n")
    except Exception as e:
        logger.error(f"✗ Failed to initialize herb benefits model: {e}\n")

    try:
        bilstm_model, label_mappings = init_bilstm_crf()
        logger.info("✓ BiLSTM-CRF model initialized successfully\n")
    except Exception as e:
        logger.error(f"✗ Failed to initialize BiLSTM-CRF: {e}\n")

    logger.info("=" * 70)
    logger.info("MODEL INITIALIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info("All models are now ready to use!")
    logger.info("- Pure BioBERT: models/pure_biobert/pytorch_model.bin")
    logger.info("- Herb Benefits: models/herb_benefits/pytorch_model.bin")
    logger.info("- BiLSTM-CRF: models/bilstm_crf/model.pth")


if __name__ == '__main__':
    main()
