"""
Cross-Domain Mapping Service for Allopathic-Ayurvedic Medicine Equivalents

This service extracts and manages mappings between allopathic medicines and their
Ayurvedic equivalents from the AyurGenixAI dataset.
"""

import csv
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import re

logger = logging.getLogger(__name__)

@dataclass
class CrossDomainMapping:
    """Represents a mapping between allopathic and Ayurvedic medicines."""
    allopathic_medicine: str
    ayurvedic_herbs: List[str]
    disease_context: str
    formulation: str = ""
    dosage_info: str = ""
    confidence_score: float = 1.0
    source: str = "ayurgenix_dataset"

@dataclass
class MappingQuery:
    """Query for cross-domain mappings."""
    query_text: str
    query_type: str = "allopathic_to_ayurvedic"  # or "ayurvedic_to_allopathic"
    include_context: bool = True

@dataclass
class MappingResponse:
    """Response containing cross-domain mappings."""
    mappings: List[CrossDomainMapping]
    query_type: str
    total_found: int
    confidence_score: float
    suggestions: List[str] = field(default_factory=list)

class CrossDomainMapper:
    """
    Service for managing cross-domain mappings between allopathic and Ayurvedic medicines.
    
    Extracts mappings from the AyurGenixAI dataset and provides query functionality
    for finding equivalent medicines across domains.
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """Initialize the cross-domain mapper."""
        self.dataset_path = dataset_path or self._get_dataset_path()
        self.mappings: List[CrossDomainMapping] = []
        self.allopathic_index: Dict[str, List[CrossDomainMapping]] = {}
        self.ayurvedic_index: Dict[str, List[CrossDomainMapping]] = {}
        self.loaded = False
        
    def _get_dataset_path(self) -> str:
        """Get the path to the AyurGenixAI dataset."""
        base_dir = Path(__file__).parent.parent.parent
        dataset_path = base_dir / "data" / "ayurgenix_dataset.csv"
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"AyurGenixAI dataset not found at: {dataset_path}")
        
        return str(dataset_path)
    
    def load_mappings(self) -> bool:
        """Load cross-domain mappings from the dataset."""
        if self.loaded:
            return True
            
        try:
            logger.info(f"Loading cross-domain mappings from: {self.dataset_path}")
            
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    mappings = self._extract_mapping_from_row(row)
                    if mappings:
                        for mapping in mappings:
                            self.mappings.append(mapping)
                            self._index_mapping(mapping)
            
            logger.info(f"Loaded {len(self.mappings)} cross-domain mappings")
            self.loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cross-domain mappings: {str(e)}")
            return False
    
    def _extract_mapping_from_row(self, row: Dict[str, str]) -> Optional[List[CrossDomainMapping]]:
        """Extract cross-domain mappings from a dataset row. Returns list of mappings."""
        try:
            disease = row.get('Disease', '').strip()
            current_medications = row.get('Current Medications', '').strip()
            ayurvedic_herbs = row.get('Ayurvedic Herbs', '').strip()
            formulation = row.get('Formulation', '').strip()
            
            # Skip if no medications or herbs specified
            if not current_medications or not ayurvedic_herbs:
                return None
            
            # Skip generic entries
            if current_medications.lower() in ['none', 'n/a', 'not applicable', '']:
                return None
            
            if ayurvedic_herbs.lower() in ['none', 'n/a', 'not applicable', '']:
                return None
            
            # Parse allopathic medicines (comma-separated)
            allopathic_medicines = [med.strip() for med in current_medications.split(',')]
            
            # Parse Ayurvedic herbs (comma-separated)
            herbs = [herb.strip() for herb in ayurvedic_herbs.split(',')]
            
            # Create mappings for each allopathic medicine
            mappings = []
            for medicine in allopathic_medicines:
                if medicine and len(medicine) > 2:
                    mapping = CrossDomainMapping(
                        allopathic_medicine=medicine,
                        ayurvedic_herbs=herbs,
                        disease_context=disease,
                        formulation=formulation,
                        confidence_score=0.9  # High confidence for dataset entries
                    )
                    mappings.append(mapping)
            
            return mappings if mappings else None
            
        except Exception as e:
            logger.warning(f"Failed to extract mapping from row: {str(e)}")
            return None
    
    def _index_mapping(self, mapping: CrossDomainMapping):
        """Index a mapping for fast lookup."""
        # Index by allopathic medicine
        allopathic_key = mapping.allopathic_medicine.lower()
        if allopathic_key not in self.allopathic_index:
            self.allopathic_index[allopathic_key] = []
        self.allopathic_index[allopathic_key].append(mapping)
        
        # Index by Ayurvedic herbs
        for herb in mapping.ayurvedic_herbs:
            herb_key = herb.lower()
            if herb_key not in self.ayurvedic_index:
                self.ayurvedic_index[herb_key] = []
            self.ayurvedic_index[herb_key].append(mapping)
    
    def query_mappings(self, query: MappingQuery) -> MappingResponse:
        """Query for cross-domain mappings."""
        if not self.loaded:
            self.load_mappings()
        
        query_text = query.query_text.lower().strip()
        
        # Extract medicine name from query
        medicine_name = self._extract_medicine_name(query_text)
        
        if query.query_type == "allopathic_to_ayurvedic":
            mappings = self._find_ayurvedic_equivalents(medicine_name)
        else:
            mappings = self._find_allopathic_equivalents(medicine_name)
        
        # Calculate overall confidence
        confidence_score = self._calculate_query_confidence(mappings, medicine_name)
        
        # Generate suggestions if no exact matches
        suggestions = []
        if not mappings:
            suggestions = self._generate_suggestions(medicine_name, query.query_type)
        
        return MappingResponse(
            mappings=mappings,
            query_type=query.query_type,
            total_found=len(mappings),
            confidence_score=confidence_score,
            suggestions=suggestions
        )
    
    def _extract_medicine_name(self, query_text: str) -> str:
        """Extract medicine name from query text."""
        # Remove common query phrases
        patterns_to_remove = [
            r'ayurvedic\s+equivalent\s+of\s+',
            r'natural\s+alternative\s+to\s+',
            r'herbal\s+substitute\s+for\s+',
            r'what\s+is\s+the\s+ayurvedic\s+',
            r'ayurvedic\s+medicine\s+for\s+',
            r'allopathic\s+equivalent\s+of\s+',
            r'modern\s+medicine\s+for\s+',
            r'western\s+medicine\s+for\s+'
        ]
        
        medicine_name = query_text
        for pattern in patterns_to_remove:
            medicine_name = re.sub(pattern, '', medicine_name, flags=re.IGNORECASE)
        
        # Clean up
        medicine_name = medicine_name.strip()
        medicine_name = re.sub(r'[^\w\s-]', '', medicine_name)  # Remove special chars except hyphens
        
        return medicine_name
    
    def _find_ayurvedic_equivalents(self, medicine_name: str) -> List[CrossDomainMapping]:
        """Find Ayurvedic equivalents for an allopathic medicine."""
        mappings = []
        
        # Direct lookup
        if medicine_name in self.allopathic_index:
            mappings.extend(self.allopathic_index[medicine_name])
        
        # Fuzzy matching
        for indexed_medicine, indexed_mappings in self.allopathic_index.items():
            if self._is_similar_medicine(medicine_name, indexed_medicine):
                mappings.extend(indexed_mappings)
        
        # Remove duplicates
        seen = set()
        unique_mappings = []
        for mapping in mappings:
            key = (mapping.allopathic_medicine, tuple(mapping.ayurvedic_herbs), mapping.disease_context)
            if key not in seen:
                seen.add(key)
                unique_mappings.append(mapping)
        
        return unique_mappings
    
    def _find_allopathic_equivalents(self, herb_name: str) -> List[CrossDomainMapping]:
        """Find allopathic equivalents for an Ayurvedic herb."""
        mappings = []
        
        # Direct lookup
        if herb_name in self.ayurvedic_index:
            mappings.extend(self.ayurvedic_index[herb_name])
        
        # Fuzzy matching
        for indexed_herb, indexed_mappings in self.ayurvedic_index.items():
            if self._is_similar_medicine(herb_name, indexed_herb):
                mappings.extend(indexed_mappings)
        
        # Remove duplicates
        seen = set()
        unique_mappings = []
        for mapping in mappings:
            key = (mapping.allopathic_medicine, tuple(mapping.ayurvedic_herbs), mapping.disease_context)
            if key not in seen:
                seen.add(key)
                unique_mappings.append(mapping)
        
        return unique_mappings
    
    def _is_similar_medicine(self, query_name: str, indexed_name: str) -> bool:
        """Check if two medicine names are similar."""
        # Simple similarity check
        if query_name in indexed_name or indexed_name in query_name:
            return True
        
        # Check for common abbreviations and variations
        query_words = set(query_name.split())
        indexed_words = set(indexed_name.split())
        
        # If any word matches, consider similar
        if query_words.intersection(indexed_words):
            return True
        
        return False
    
    def _calculate_query_confidence(self, mappings: List[CrossDomainMapping], medicine_name: str) -> float:
        """Calculate confidence score for query results."""
        if not mappings:
            return 0.0
        
        # Base confidence on number of mappings and exact matches
        exact_matches = sum(1 for m in mappings if medicine_name in m.allopathic_medicine.lower())
        total_mappings = len(mappings)
        
        if exact_matches > 0:
            return min(0.9, 0.7 + (exact_matches / total_mappings) * 0.2)
        else:
            return min(0.6, 0.4 + (total_mappings / 10) * 0.2)
    
    def _generate_suggestions(self, medicine_name: str, query_type: str) -> List[str]:
        """Generate suggestions when no exact matches are found."""
        suggestions = []
        
        if query_type == "allopathic_to_ayurvedic":
            # Suggest similar allopathic medicines
            for indexed_medicine in self.allopathic_index.keys():
                if self._is_similar_medicine(medicine_name, indexed_medicine):
                    suggestions.append(f"Did you mean '{indexed_medicine}'?")
                    if len(suggestions) >= 3:
                        break
        else:
            # Suggest similar Ayurvedic herbs
            for indexed_herb in self.ayurvedic_index.keys():
                if self._is_similar_medicine(medicine_name, indexed_herb):
                    suggestions.append(f"Did you mean '{indexed_herb}'?")
                    if len(suggestions) >= 3:
                        break
        
        return suggestions
    
    def get_all_allopathic_medicines(self) -> List[str]:
        """Get list of all allopathic medicines in the database."""
        if not self.loaded:
            self.load_mappings()
        return list(self.allopathic_index.keys())
    
    def get_all_ayurvedic_herbs(self) -> List[str]:
        """Get list of all Ayurvedic herbs in the database."""
        if not self.loaded:
            self.load_mappings()
        return list(self.ayurvedic_index.keys())
    
    def get_mapping_stats(self) -> Dict[str, Any]:
        """Get statistics about the mappings."""
        if not self.loaded:
            self.load_mappings()
        
        return {
            "total_mappings": len(self.mappings),
            "unique_allopathic_medicines": len(self.allopathic_index),
            "unique_ayurvedic_herbs": len(self.ayurvedic_index),
            "diseases_covered": len(set(m.disease_context for m in self.mappings)),
            "average_herbs_per_medicine": sum(len(m.ayurvedic_herbs) for m in self.mappings) / len(self.mappings) if self.mappings else 0
        }

# Global instance for reuse
_global_cross_domain_mapper = None

def get_cross_domain_mapper() -> CrossDomainMapper:
    """Get or create global cross-domain mapper instance."""
    global _global_cross_domain_mapper
    if _global_cross_domain_mapper is None:
        _global_cross_domain_mapper = CrossDomainMapper()
    return _global_cross_domain_mapper