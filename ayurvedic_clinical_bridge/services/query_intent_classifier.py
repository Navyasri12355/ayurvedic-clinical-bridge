"""
Query Intent Classifier Service

Classifies queries into herb_benefits, disease_prediction, or general_info
using a trained BERT-based model exclusively. No keyword lists or rule-based
fallbacks are used — if the model is unavailable the classifier returns
'general_info' with low confidence and logs a warning.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import json
from pathlib import Path
import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

# Default label map (overwritten when a trained model is loaded)
DEFAULT_LABEL_MAPPING: Dict[str, str] = {
    '0': 'disease_prediction',
    '1': 'general_info',
    '2': 'herb_benefits',
}


class QueryIntentClassifier(nn.Module):
    """BERT-based query intent classifier."""

    def __init__(self, model_name: str = 'dmis-lab/biobert-v1.1', num_classes: int = 3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.pooler_output))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_classes), labels.view(-1))
        return {'loss': loss, 'logits': logits} if loss is not None else logits


class QueryIntentClassifierService:
    """
    Service for classifying query intent using a trained ML model.

    The *only* classification path is through the trained model.  There is no
    keyword-based fallback — degraded behaviour is an explicit 'general_info'
    response with confidence 0.0 so callers can detect the failure.
    """

    def __init__(self):
        self.model: QueryIntentClassifier | None = None
        self.tokenizer = None
        self.label_mapping: Dict[str, str] = dict(DEFAULT_LABEL_MAPPING)
        self.is_loaded = False
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        model_path = Path('models/query_intent_classifier')
        if not model_path.exists():
            logger.warning("Query intent classifier model not found at %s", model_path)
            return

        try:
            # Label mapping
            lm_path = model_path.parent / 'query_intent_label_mapping.json'
            if lm_path.exists():
                self.label_mapping = json.loads(lm_path.read_text())

            # Config
            cfg_path = model_path / 'config.json'
            num_classes, model_name = 3, 'dmis-lab/biobert-v1.1'
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text())
                num_classes = cfg.get('num_classes', num_classes)
                model_name = cfg.get('model_name', model_name)

            # Tokenizer + weights
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = QueryIntentClassifier(model_name, num_classes)
            weights_path = model_path / 'pytorch_model.bin'
            if not weights_path.exists():
                logger.warning("Model weights not found at %s", weights_path)
                return
            self.model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            self.model.eval()
            self.is_loaded = True
            logger.info("Query intent classifier loaded successfully")
        except Exception as e:
            logger.error("Failed to load query intent classifier: %s", e)
            self.model = None
            self.tokenizer = None
            self.is_loaded = False

    # ------------------------------------------------------------------
    # Inference — model only, no rules
    # ------------------------------------------------------------------

    def classify_intent(self, query: str) -> Tuple[str, float]:
        """
        Classify query intent using the trained model.

        Returns ('general_info', 0.0) when the model is not available.
        """
        if not self.is_loaded or not self.model or not self.tokenizer:
            logger.warning("Intent classifier not loaded — defaulting to general_info")
            return 'general_info', 0.0

        try:
            enc = self.tokenizer(
                query,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt',
            )
            with torch.no_grad():
                output = self.model(enc['input_ids'], enc['attention_mask'])
                logits = output['logits'] if isinstance(output, dict) else output
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs.max().item()
            intent = self.label_mapping.get(str(pred), 'general_info')
            return intent, confidence
        except Exception as e:
            logger.error("Error during intent classification: %s", e)
            return 'general_info', 0.0

    def get_intent_probabilities(self, query: str) -> Dict[str, float]:
        """Return per-class probabilities from the model."""
        if not self.is_loaded or not self.model or not self.tokenizer:
            return {label: 0.0 for label in self.label_mapping.values()}

        try:
            enc = self.tokenizer(
                query,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt',
            )
            with torch.no_grad():
                output = self.model(enc['input_ids'], enc['attention_mask'])
                logits = output['logits'] if isinstance(output, dict) else output
                probs = torch.softmax(logits, dim=1).squeeze()
            return {
                self.label_mapping.get(str(i), f'class_{i}'): p.item()
                for i, p in enumerate(probs)
            }
        except Exception as e:
            logger.error("Error computing intent probabilities: %s", e)
            return {label: 0.0 for label in self.label_mapping.values()}

    def is_available(self) -> bool:
        return self.is_loaded


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_intent_classifier_service: QueryIntentClassifierService | None = None


def get_intent_classifier_service() -> QueryIntentClassifierService:
    global _intent_classifier_service
    if _intent_classifier_service is None:
        _intent_classifier_service = QueryIntentClassifierService()
    return _intent_classifier_service