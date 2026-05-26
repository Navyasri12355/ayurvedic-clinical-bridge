"""
BiLSTM-CRF Model for Clinical Named Entity Recognition

This module implements a BiLSTM-CRF architecture optimized for fast and accurate
clinical entity recognition, particularly suitable for general users requiring
quick responses with good accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.
    
    This implementation provides efficient CRF inference and training
    for clinical entity recognition tasks.
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        Initialize CRF layer.
        
        Args:
            num_tags: Number of entity tags
            batch_first: Whether input tensors are batch-first
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # Transition parameters: transitions[i][j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # Start and end transitions
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
    
    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute CRF loss.
        
        Args:
            emissions: (batch_size, seq_len, num_tags) - emission scores
            tags: (batch_size, seq_len) - true tag sequence
            mask: (batch_size, seq_len) - mask for valid positions
            
        Returns:
            Negative log-likelihood loss
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        # Compute log partition function
        log_partition = self._compute_log_partition(emissions, mask)
        
        # Compute score of true sequence
        gold_score = self._compute_score(emissions, tags, mask)
        
        # Return negative log-likelihood
        return log_partition - gold_score
    
    def decode(self, emissions: torch.Tensor, 
               mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        Viterbi decoding to find best tag sequence.
        
        Args:
            emissions: (batch_size, seq_len, num_tags) - emission scores
            mask: (batch_size, seq_len) - mask for valid positions
            
        Returns:
            List of best tag sequences for each batch item
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        # Ensure mask is boolean
        if mask.dtype != torch.bool:
            mask = mask.bool()
        
        batch_size, seq_len = emissions.shape[:2]
        
        # Viterbi algorithm
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch_size, num_tags)
        history = []
        
        for i in range(1, seq_len):
            # Broadcast score and transitions
            broadcast_score = score.unsqueeze(2)  # (batch_size, num_tags, 1)
            broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            
            # Compute next scores
            next_score = broadcast_score + broadcast_transitions + emissions[:, i].unsqueeze(1)
            
            # Find best previous tags
            next_score, indices = next_score.max(dim=1)
            
            # Apply mask
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
            history.append(indices)
        
        # Add end transitions
        score += self.end_transitions.unsqueeze(0)
        
        # Backtrack
        best_tags_list = []
        for batch_idx in range(batch_size):
            # Find best final tag
            seq_len_batch = mask[batch_idx].sum().item()
            best_last_tag = score[batch_idx].argmax().item()
            
            # Backtrack through history
            best_tags = [best_last_tag]
            for hist_idx in reversed(range(len(history))):
                if hist_idx < seq_len_batch - 1:
                    best_last_tag = history[hist_idx][batch_idx, best_last_tag].item()
                    best_tags.append(best_last_tag)
            
            # Reverse to get forward sequence and truncate to actual length
            best_tags.reverse()
            best_tags_list.append(best_tags[:seq_len_batch])
        
        return best_tags_list
    
    def _compute_log_partition(self, emissions: torch.Tensor, 
                              mask: torch.Tensor) -> torch.Tensor:
        """Compute log partition function using forward algorithm."""
        batch_size, seq_len = emissions.shape[:2]
        
        # Initialize with start transitions
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (batch_size, num_tags)
        
        for i in range(1, seq_len):
            # Broadcast for transition computation
            broadcast_score = score.unsqueeze(2)  # (batch_size, num_tags, 1)
            broadcast_transitions = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            broadcast_emissions = emissions[:, i].unsqueeze(1)  # (batch_size, 1, num_tags)
            
            # Compute next scores
            next_score = broadcast_score + broadcast_transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)  # (batch_size, num_tags)
            
            # Apply mask
            score = torch.where(mask[:, i].unsqueeze(1), next_score, score)
        
        # Add end transitions and sum over all possible end tags
        score += self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)  # (batch_size,)
    
    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, 
                      mask: torch.Tensor) -> torch.Tensor:
        """Compute score of given tag sequence."""
        batch_size, seq_len = emissions.shape[:2]
        
        # Create a mask for valid tags (not -100)
        valid_mask = (tags != -100) & mask
        
        # Replace -100 with 0 for indexing (will be masked out anyway)
        safe_tags = tags.clone()
        safe_tags[tags == -100] = 0
        
        # Emission scores
        emission_scores = emissions.gather(2, safe_tags.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)
        emission_scores = emission_scores * valid_mask.float()
        
        # Start transition scores (only for first valid tag)
        start_scores = torch.zeros(batch_size, device=emissions.device)
        for i in range(batch_size):
            first_valid = valid_mask[i].nonzero(as_tuple=False)
            if len(first_valid) > 0:
                first_idx = first_valid[0].item()
                start_scores[i] = self.start_transitions[safe_tags[i, first_idx]]
        
        # Transition scores
        transition_scores = torch.zeros(batch_size, device=emissions.device)
        for i in range(1, seq_len):
            prev_valid = valid_mask[:, i-1]
            curr_valid = valid_mask[:, i]
            both_valid = prev_valid & curr_valid
            
            if both_valid.any():
                prev_tags = safe_tags[:, i-1][both_valid]
                curr_tags = safe_tags[:, i][both_valid]
                trans_scores = self.transitions[prev_tags, curr_tags]
                
                # Add to corresponding batch items
                batch_indices = both_valid.nonzero(as_tuple=False).squeeze(1)
                transition_scores[batch_indices] += trans_scores
        
        # End transition scores (only for last valid tag)
        end_scores = torch.zeros(batch_size, device=emissions.device)
        for i in range(batch_size):
            valid_positions = valid_mask[i].nonzero(as_tuple=False)
            if len(valid_positions) > 0:
                last_idx = valid_positions[-1].item()
                end_scores[i] = self.end_transitions[safe_tags[i, last_idx]]
        
        return emission_scores.sum(dim=1) + start_scores + transition_scores + end_scores


@dataclass
class BiLSTMCRFConfig:
    """Configuration for BiLSTM-CRF model."""
    vocab_size: int = 30000
    embedding_dim: int = 300
    hidden_dim: int = 256
    num_layers: int = 2
    num_labels: int = 19  
    dropout: float = 0.3
    bidirectional: bool = True
    use_pretrained_embeddings: bool = True
    pretrained_embedding_path: Optional[str] = None
    freeze_embeddings: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_length: int = 256
    batch_size: int = 32
    eos_token_id: int = 2
    pad_token_id: int = 0
    bos_token_id: int = 1


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for clinical named entity recognition.
    
    This model combines bidirectional LSTM for context modeling
    with CRF for structured prediction, optimized for speed
    and accuracy in clinical entity recognition tasks.
    """
    
    def __init__(self, config: BiLSTMCRFConfig, vocab_to_idx: Optional[Dict[str, int]] = None):
        """
        Initialize BiLSTM-CRF model.
        
        Args:
            config: Model configuration
            vocab_to_idx: Vocabulary mapping (optional)
        """
        super(BiLSTMCRF, self).__init__()
        self.config = config
        self.vocab_to_idx = vocab_to_idx or {}
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
        
        # Embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.embedding_dim,
            padding_idx=config.pad_token_id
        )
        
        if config.freeze_embeddings:
            self.embedding.weight.requires_grad = False
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Linear layer to map LSTM output to tag space
        lstm_output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.hidden2tag = nn.Linear(lstm_output_dim, config.num_labels)
        
        # CRF layer
        self.crf = CRF(config.num_labels)
        
        logger.info(f"Initialized BiLSTM-CRF with {self._count_parameters()} parameters")
    
    def _count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of BiLSTM-CRF model.
        
        Args:
            input_ids: (batch_size, seq_len) - input token ids
            attention_mask: (batch_size, seq_len) - attention mask
            labels: (batch_size, seq_len) - true labels (for training)
            
        Returns:
            Dictionary containing loss (if labels provided) and predictions
        """
        batch_size, seq_len = input_ids.shape
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        
        # Ensure attention_mask is boolean
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # Embedding
        embeddings = self.embedding(input_ids)  # (batch_size, seq_len, embedding_dim)
        embeddings = self.dropout(embeddings)
        
        # Pack padded sequences for efficient LSTM processing
        seq_lengths = attention_mask.sum(dim=1).cpu()
        
        # Ensure minimum sequence length of 1
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM
        packed_lstm_out, _ = self.lstm(packed_embeddings)
        lstm_out, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_lstm_out, batch_first=True, total_length=seq_len
        )
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Map to tag space
        emissions = self.hidden2tag(lstm_out)  # (batch_size, seq_len, num_labels)
        
        outputs_dict = {"logits": emissions}
        
        if labels is not None:
            # Filter out padding tokens from labels
            valid_labels = labels.clone()
            valid_labels[~attention_mask] = -100
            
            # Compute CRF loss
            loss = self.crf(emissions, valid_labels, attention_mask)
            outputs_dict["loss"] = loss.mean()
            
            # Decode best sequence
            predictions = self.crf.decode(emissions, attention_mask)
            # Convert to tensor format for consistency
            pred_tensor = torch.zeros_like(labels)
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=labels.device)
            outputs_dict["predictions"] = pred_tensor
        else:
            # Decode best sequence
            predictions = self.crf.decode(emissions, attention_mask)
            # Convert to tensor format
            pred_tensor = torch.zeros(batch_size, seq_len, dtype=torch.long, device=input_ids.device)
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=input_ids.device)
            outputs_dict["predictions"] = pred_tensor
        
        return outputs_dict
    
    def load_pretrained_embeddings(self, embedding_path: str):
        """Load pre-trained word embeddings."""
        try:
            # This would load embeddings like GloVe, Word2Vec, etc.
            # For now, we'll use random initialization
            logger.info(f"Loading pretrained embeddings from {embedding_path}")
            # Implementation would depend on embedding format
            pass
        except Exception as e:
            logger.warning(f"Failed to load pretrained embeddings: {e}")


class BiLSTMCRFTokenizer:
    """
    Simple tokenizer for BiLSTM-CRF model.
    
    This tokenizer provides basic word-level tokenization
    optimized for clinical text processing.
    """
    
    def __init__(self, vocab_to_idx: Optional[Dict[str, int]] = None):
        """Initialize tokenizer with vocabulary."""
        self.vocab_to_idx = vocab_to_idx or {}
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        
        # Add special tokens to vocabulary if not present
        special_tokens = [self.pad_token, self.unk_token, self.cls_token, self.sep_token]
        for token in special_tokens:
            if token not in self.vocab_to_idx:
                self.vocab_to_idx[token] = len(self.vocab_to_idx)
        
        self.idx_to_vocab = {v: k for k, v in self.vocab_to_idx.items()}
        
        # Token IDs
        self.pad_token_id = self.vocab_to_idx[self.pad_token]
        self.unk_token_id = self.vocab_to_idx[self.unk_token]
        self.cls_token_id = self.vocab_to_idx[self.cls_token]
        self.sep_token_id = self.vocab_to_idx[self.sep_token]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        # Simple word tokenization
        import re
        # Split on whitespace and punctuation, but keep medical terms intact
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return tokens
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs."""
        return [self.vocab_to_idx.get(token, self.unk_token_id) for token in tokens]
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens."""
        return [self.idx_to_vocab.get(id, self.unk_token) for id in ids]
    
    def encode(self, text: str, max_length: int = 256, padding: bool = True, 
               truncation: bool = True, return_tensors: Optional[str] = None) -> Dict[str, Any]:
        """
        Encode text to model inputs.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            return_tensors: Format of returned tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        tokens = self.tokenize(text)
        
        # Add special tokens
        tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.sep_token]
        
        # Convert to IDs
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad if necessary
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            import torch
            result = {k: torch.tensor([v]) for k, v in result.items()}
        
        return result
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        tokens = self.convert_ids_to_tokens(token_ids)
        
        if skip_special_tokens:
            special_tokens = {self.pad_token, self.unk_token, self.cls_token, self.sep_token}
            tokens = [token for token in tokens if token not in special_tokens]
        
        return ' '.join(tokens)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Load tokenizer from saved model directory."""
        vocab_path = Path(model_path) / "vocab.json"
        if vocab_path.exists():
            import json
            with open(vocab_path, 'r') as f:
                vocab_to_idx = json.load(f)
            return cls(vocab_to_idx)
        else:
            logger.warning(f"No vocabulary found at {vocab_path}, using default tokenizer")
            return cls()
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save vocabulary
        import json
        with open(save_path / "vocab.json", 'w') as f:
            json.dump(self.vocab_to_idx, f, indent=2)
        
        logger.info(f"Tokenizer saved to {save_directory}")


def create_default_vocab() -> Dict[str, int]:
    """Create a default vocabulary for clinical text."""
    # Common clinical and Ayurvedic terms
    clinical_terms = [
        # Medical conditions
        'diabetes', 'hypertension', 'arthritis', 'asthma', 'migraine',
        'anxiety', 'depression', 'insomnia', 'obesity', 'fever',
        'cough', 'cold', 'indigestion', 'constipation', 'acidity',
        
        # Symptoms
        'pain', 'inflammation', 'swelling', 'headache', 'nausea',
        'fatigue', 'dizziness', 'burning', 'stiffness', 'weakness',
        'thirst', 'urination', 'vision', 'blurred',
        
        # Ayurvedic herbs
        'turmeric', 'ginger', 'ashwagandha', 'brahmi', 'gudmar',
        'arjuna', 'tulsi', 'neem', 'amla', 'triphala', 'guggulu',
        'shatavari', 'licorice', 'fenugreek', 'cinnamon', 'cardamom',
        
        # Dosage and administration
        'mg', 'gram', 'grams', 'tsp', 'teaspoon', 'tbsp', 'tablespoon',
        'daily', 'twice', 'three', 'times', 'capsule', 'tablet',
        'powder', 'extract', 'oil', 'juice',
        
        # Common words
        'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for',
        'with', 'by', 'from', 'up', 'about', 'into', 'through',
        'during', 'before', 'after', 'above', 'below', 'between',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our',
        'their', 'this', 'that', 'these', 'those', 'am', 'is',
        'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can'
    ]
    
    # Create vocabulary with special tokens first
    vocab_to_idx = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3
    }
    
    # Add clinical terms
    for term in clinical_terms:
        if term not in vocab_to_idx:
            vocab_to_idx[term] = len(vocab_to_idx)
    
    return vocab_to_idx