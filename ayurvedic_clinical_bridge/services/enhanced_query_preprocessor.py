#!/usr/bin/env python3
"""
Enhanced Query Preprocessor for Pure ML System
Normalizes herb names and query patterns for improved tokenization and classification
"""

import json
import re
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd

class EnhancedQueryPreprocessor:
    """
    Enhanced query preprocessor that normalizes herb names and query patterns
    for improved ML model performance without using hardcoded rules.
    """
    
    def __init__(self):
        """Initialize the preprocessor with herb normalization mappings."""
        self.herb_normalization_rules = {}
        self.query_patterns = {}
        self.load_herb_normalization_rules()
        self.load_query_patterns()
    
    def load_herb_normalization_rules(self) -> Dict[str, str]:
        """Load herb name normalization mappings from training data."""
        try:
            # Load from Amidha database
            amidha_path = Path('data/amidha_herbs_comprehensive.json')
            if amidha_path.exists():
                with open(amidha_path, 'r', encoding='utf-8') as f:
                    herbs_data = json.load(f)
                
                # Create normalization rules from herb database
                for herb in herbs_data:
                    name = herb.get('name', '').lower().strip()
                    if name:
                        # Add the canonical name
                        self.herb_normalization_rules[name] = name
                        
                        # Add common variations
                        variations = self._generate_herb_variations(name)
                        for variation in variations:
                            self.herb_normalization_rules[variation] = name
            
            # Add common manual mappings for frequently misspelled herbs
            common_mappings = {
                'ginjer': 'ginger',
                'gingr': 'ginger',
                'tumeric': 'turmeric',
                'tumric': 'turmeric',
                'holy basil': 'tulsi',
                'indian basil': 'tulsi',
                'ashwaganda': 'ashwagandha',
                'winter cherry': 'ashwagandha',
                'brahmi': 'brahmi',
                'bacopa': 'brahmi',
                'shankhpushpi': 'shankhpushpi',
                'conch shell': 'shankhpushpi',
                'guduchi': 'guduchi',
                'giloy': 'guduchi',
                'triphala': 'triphala',
                'three fruits': 'triphala'
            }
            
            self.herb_normalization_rules.update(common_mappings)
            
            print(f"✅ Loaded {len(self.herb_normalization_rules)} herb normalization rules")
            return self.herb_normalization_rules
            
        except Exception as e:
            print(f"❌ Error loading herb normalization rules: {e}")
            return {}
    
    def _generate_herb_variations(self, herb_name: str) -> List[str]:
        """Generate common variations of herb names."""
        variations = []
        
        # Remove spaces and hyphens
        variations.append(herb_name.replace(' ', ''))
        variations.append(herb_name.replace('-', ''))
        variations.append(herb_name.replace(' ', '-'))
        
        # Common misspellings (simple character substitutions)
        if 'ph' in herb_name:
            variations.append(herb_name.replace('ph', 'f'))
        if 'gh' in herb_name:
            variations.append(herb_name.replace('gh', 'g'))
        
        # Plural forms
        if not herb_name.endswith('s'):
            variations.append(herb_name + 's')
        
        return list(set(variations))
    
    def load_query_patterns(self) -> Dict[str, str]:
        """Load query pattern normalization rules."""
        # Common query pattern normalizations
        self.query_patterns = {
            r'\bwhat are the benefits of\b': 'benefits of',
            r'\bwhat is .+ good for\b': lambda m: f"benefits of {m.group().split()[2]}",
            r'\bhow does .+ help\b': lambda m: f"benefits of {m.group().split()[2]}",
            r'\btell me about\b': 'benefits of',
            r'\binformation about\b': 'benefits of',
            r'\bproperties of\b': 'benefits of',
            r'\badvantages of\b': 'benefits of',
            r'\btherapeutic effects of\b': 'benefits of',
            r'\bmedicinal properties of\b': 'benefits of',
            r'\bhealth benefits of\b': 'benefits of',
            r'\buses of\b': 'benefits of',
            r'\bwhat does .+ do\b': lambda m: f"benefits of {m.group().split()[2]}",
            r'\bwhy use\b': 'benefits of'
        }
        
        print(f"✅ Loaded {len(self.query_patterns)} query pattern rules")
        return self.query_patterns
    
    def normalize_herb_names(self, text: str) -> str:
        """Normalize herb names in the text using learned mappings."""
        normalized_text = text.lower()
        
        # Sort by length (longest first) to avoid partial replacements
        sorted_herbs = sorted(self.herb_normalization_rules.keys(), key=len, reverse=True)
        
        for variation, canonical in [(h, self.herb_normalization_rules[h]) for h in sorted_herbs]:
            if variation in normalized_text:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(variation) + r'\b'
                normalized_text = re.sub(pattern, canonical, normalized_text)
        
        return normalized_text
    
    def normalize_query_patterns(self, text: str) -> str:
        """Normalize query patterns for consistent processing."""
        normalized_text = text.lower()
        
        for pattern, replacement in self.query_patterns.items():
            if callable(replacement):
                normalized_text = re.sub(pattern, replacement, normalized_text)
            else:
                normalized_text = re.sub(pattern, replacement, normalized_text)
        
        return normalized_text
    
    def preprocess_query(self, text: str) -> str:
        """
        Main preprocessing function that normalizes herb names and query patterns.
        
        Args:
            text: Raw user query
            
        Returns:
            Normalized query text ready for tokenization
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Step 1: Basic cleaning
        cleaned_text = text.strip().lower()
        
        # Step 2: Normalize herb names
        herb_normalized = self.normalize_herb_names(cleaned_text)
        
        # Step 3: Normalize query patterns
        pattern_normalized = self.normalize_query_patterns(herb_normalized)
        
        # Step 4: Clean up extra spaces
        final_text = re.sub(r'\s+', ' ', pattern_normalized).strip()
        
        return final_text
    
    def extract_herb_entities(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract herb entities from text with their positions.
        
        Args:
            text: Input text
            
        Returns:
            List of tuples (herb_name, start_pos, end_pos)
        """
        entities = []
        normalized_text = text.lower()
        
        # Sort by length (longest first) to avoid partial matches
        sorted_herbs = sorted(self.herb_normalization_rules.values(), key=len, reverse=True)
        unique_herbs = list(set(sorted_herbs))
        
        for herb in unique_herbs:
            pattern = r'\b' + re.escape(herb) + r'\b'
            matches = re.finditer(pattern, normalized_text)
            
            for match in matches:
                entities.append((herb, match.start(), match.end()))
        
        # Remove overlapping entities (keep longest)
        entities = self._remove_overlapping_entities(entities)
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Tuple[str, int, int]]) -> List[Tuple[str, int, int]]:
        """Remove overlapping entities, keeping the longest ones."""
        if not entities:
            return []
        
        # Sort by start position, then by length (descending)
        entities.sort(key=lambda x: (x[1], -(x[2] - x[1])))
        
        non_overlapping = []
        for entity in entities:
            herb, start, end = entity
            
            # Check if this entity overlaps with any already selected
            overlaps = False
            for selected_herb, selected_start, selected_end in non_overlapping:
                if not (end <= selected_start or start >= selected_end):
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        return non_overlapping
    
    def get_preprocessing_stats(self) -> Dict[str, int]:
        """Get statistics about the preprocessing rules."""
        return {
            'herb_normalization_rules': len(self.herb_normalization_rules),
            'query_patterns': len(self.query_patterns),
            'unique_canonical_herbs': len(set(self.herb_normalization_rules.values()))
        }

# Example usage and testing
if __name__ == "__main__":
    preprocessor = EnhancedQueryPreprocessor()
    
    # Test cases
    test_queries = [
        "benefits of ginjer",  # Should normalize to "benefits of ginger"
        "what are the benefits of tumeric",  # Should normalize to "benefits of turmeric"
        "tell me about holy basil",  # Should normalize to "benefits of tulsi"
        "ashwaganda uses",  # Should normalize to "ashwagandha uses"
        "what is brahmi good for",  # Should normalize to "benefits of brahmi"
        "therapeutic effects of guduchi"  # Should normalize to "benefits of guduchi"
    ]
    
    print("🧪 Testing Enhanced Query Preprocessor")
    print("=" * 50)
    
    for query in test_queries:
        normalized = preprocessor.preprocess_query(query)
        entities = preprocessor.extract_herb_entities(normalized)
        
        print(f"Original: {query}")
        print(f"Normalized: {normalized}")
        print(f"Entities: {entities}")
        print("-" * 30)
    
    # Print stats
    stats = preprocessor.get_preprocessing_stats()
    print(f"\n📊 Preprocessing Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")