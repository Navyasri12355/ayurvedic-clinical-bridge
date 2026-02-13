"""
Semantic Herb Matcher
Uses BioBERT embeddings to detect herb mentions in queries via cosine similarity.
"""

import re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SemanticHerbMatcher:
    """
    Semantic herb detection using BioBERT embeddings.
    
    Pre-computes embeddings for all known herbs, then at runtime
    compares query embeddings to find matching herbs.
    """
    
    def __init__(
        self, 
        herb_names: List[str], 
        model_name: str = 'dmis-lab/biobert-base-cased-v1.1',
        similarity_threshold: float = 0.85,  # Raised from 0.65 to be more strict
        max_matches: int = 2  # Limit to top 2 matches
    ):
        """
        Initialize the semantic herb matcher.
        
        Args:
            herb_names: List of known herb names to match against
            model_name: HuggingFace model name for embeddings
            similarity_threshold: Minimum similarity score to consider a match
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity_threshold = similarity_threshold
        self.max_matches = max_matches
        self.herb_names = herb_names
        self.herb_embeddings = None
        self.tokenizer = None
        self.model = None
        
        # Build alias mapping: maps alternative names to canonical names
        self.alias_to_canonical = {}
        self._build_alias_mapping()
        
        self._load_model(model_name)
        self._precompute_herb_embeddings()
    
    def _build_alias_mapping(self):
        """
        Build an alias mapping from English names (in parentheses) to canonical names.
        Also adds the base Sanskrit name as an alias.
        E.g., "Shunti (Ginger)" -> aliases: ["shunti", "ginger"]
        """
        for herb_name in self.herb_names:
            canonical = herb_name
            herb_lower = herb_name.lower()
            
            # Add full name as alias
            self.alias_to_canonical[herb_lower] = canonical
            
            # Extract English name from parentheses: "Karpura (Camphor)" -> "camphor"
            match = re.search(r'\(([^)]+)\)', herb_name)
            if match:
                english_name = match.group(1).lower().strip()
                self.alias_to_canonical[english_name] = canonical
                
                # Also add Sanskrit name without parentheses
                sanskrit_name = herb_name.split('(')[0].strip().lower()
                self.alias_to_canonical[sanskrit_name] = canonical
            
        logger.info(f"Built alias mapping for {len(self.alias_to_canonical)} herb names")
    
    def _load_model(self, model_name: str):
        """Load BioBERT model and tokenizer."""
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            self.tokenizer = None
    
    def _get_embedding(self, text: str) -> Optional[torch.Tensor]:
        """
        Get the embedding for a piece of text.
        Uses mean pooling of the last hidden state.
        """
        if not self.model or not self.tokenizer:
            return None
        
        try:
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over non-padding tokens
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            
            # Expand attention mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            
            # Sum embeddings and divide by number of tokens
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            
            embedding = sum_embeddings / sum_mask
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing embedding for '{text}': {e}")
            return None
    
    def _precompute_herb_embeddings(self):
        """Pre-compute embeddings for all known herbs."""
        if not self.model or not self.tokenizer:
            logger.warning("Model not loaded, cannot precompute herb embeddings")
            return
        
        logger.info(f"Pre-computing embeddings for {len(self.herb_names)} herbs...")
        
        embeddings = []
        valid_herbs = []
        
        for herb in self.herb_names:
            emb = self._get_embedding(herb)
            if emb is not None:
                embeddings.append(emb)
                valid_herbs.append(herb)
        
        if embeddings:
            self.herb_embeddings = torch.cat(embeddings, dim=0)  # Shape: (num_herbs, hidden_size)
            self.herb_names = valid_herbs
            logger.info(f"Successfully embedded {len(valid_herbs)} herbs")
        else:
            logger.error("Failed to compute any herb embeddings")
    
    def find_herbs_in_query(self, query: str, threshold: Optional[float] = None) -> List[Dict[str, any]]:
        """
        Find herbs mentioned in the query.
        
        Strategy:
        1. First check for EXACT name matches using alias mapping (most reliable)
        2. Only use semantic similarity if no exact matches found
        
        Args:
            query: User query to analyze
            threshold: Optional custom threshold for semantic matching
            
        Returns:
            List of detected herbs with similarity scores
        """
        matches = []
        query_lower = query.lower()
        found_canonicals = set()  # Track which canonical names we've already added
        
        # STEP 1: Check for exact matches using alias mapping
        # This matches both English names (ginger, turmeric) and Sanskrit names (shunti, haridra)
        for alias, canonical in self.alias_to_canonical.items():
            if alias in query_lower and canonical not in found_canonicals:
                matches.append({
                    'name': canonical,
                    'similarity': 1.0,  # Perfect match
                    'confidence': '100.0%',
                    'match_type': 'exact',
                    'matched_alias': alias
                })
                found_canonicals.add(canonical)
                logger.info(f"Exact match found: '{alias}' -> {canonical}")
        
        # If we found exact matches, return them (no need for semantic search)
        if matches:
            return matches[:self.max_matches]
        
        # STEP 2: Semantic fallback (only if no exact matches)
        if self.herb_embeddings is None or self.model is None:
            logger.warning("Semantic matcher not initialized, returning empty")
            return []
        
        if threshold is None:
            threshold = self.similarity_threshold
        
        try:
            # Get query embedding
            query_emb = self._get_embedding(query)
            if query_emb is None:
                return []
            
            # Compute cosine similarity with all herb embeddings
            similarities = F.cosine_similarity(
                query_emb,
                self.herb_embeddings,
                dim=1
            )
            
            # Find herbs above threshold
            for i, (herb, sim) in enumerate(zip(self.herb_names, similarities)):
                sim_value = sim.item()
                if sim_value >= threshold:
                    matches.append({
                        'name': herb,
                        'similarity': sim_value,
                        'confidence': f"{sim_value * 100:.1f}%",
                        'match_type': 'semantic'
                    })
            
            # Sort by similarity descending and limit
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            matches = matches[:self.max_matches]
            
            if matches:
                logger.info(f"Semantic matcher found {len(matches)} herbs: {[m['name'] for m in matches]}")
            
            return matches
            
        except Exception as e:
            logger.error(f"Error in semantic herb matching: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if the semantic matcher is ready."""
        return (
            self.model is not None and 
            self.tokenizer is not None and 
            self.herb_embeddings is not None
        )
