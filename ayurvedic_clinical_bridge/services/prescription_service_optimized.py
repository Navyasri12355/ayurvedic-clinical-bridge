"""
Optimized Prescription Processing Service for real-time entity extraction.

This service provides efficient prescription parsing using rule-based NER
with medical entity recognition patterns.
"""

import re
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class MedicalEntity:
    """Medical entity extracted from prescription text."""
    entity_type: str
    text: str
    start_pos: int
    end_pos: int
    confidence: float
    normalized_form: Optional[str] = None

@dataclass
class ParsedPrescription:
    """Parsed prescription with extracted entities."""
    original_text: str
    entities: List[MedicalEntity]
    confidence_score: float
    processing_time: float
    warnings: List[str]

class OptimizedPrescriptionService:
    """Optimized prescription processing with rule-based NER."""
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize the service."""
        self.confidence_threshold = confidence_threshold
        self._load_patterns()
        logger.info("Initialized OptimizedPrescriptionService")
    
    def _load_patterns(self):
        """Load medical entity recognition patterns."""
        # Drug name patterns (common medications)
        self.drug_patterns = [
            # Common drugs
            (r'\b(aspirin|ibuprofen|acetaminophen|paracetamol|tylenol)\b', 0.95),
            (r'\b(metformin|insulin|lisinopril|atorvastatin|amlodipine)\b', 0.95),
            (r'\b(omeprazole|simvastatin|levothyroxine|warfarin|clopidogrel)\b', 0.95),
            (r'\b(losartan|hydrochlorothiazide|gabapentin|sertraline|trazodone)\b', 0.95),
            
            # Ayurvedic herbs
            (r'\b(turmeric|haldi|haridra)\b', 0.90),
            (r'\b(ginger|adrak|shunthi)\b', 0.90),
            (r'\b(ashwagandha|withania)\b', 0.90),
            (r'\b(brahmi|bacopa)\b', 0.90),
            (r'\b(neem|margosa)\b', 0.90),
            (r'\b(triphala|amla|haritaki|bibhitaki)\b', 0.90),
            (r'\b(guduchi|giloy|tinospora)\b', 0.90),
            (r'\b(tulsi|holy basil|ocimum)\b', 0.90),
            
            # Generic drug patterns
            (r'\b[A-Z][a-z]+(?:ol|in|ine|ate|ide|ium)\b', 0.70),  # Common drug suffixes
            (r'\b[A-Z][a-z]*[A-Z][a-z]*\b', 0.60),  # CamelCase drug names
        ]
        
        # Dosage patterns
        self.dosage_patterns = [
            (r'\b(\d+(?:\.\d+)?)\s*(mg|milligrams?|g|grams?|mcg|micrograms?)\b', 0.95),
            (r'\b(\d+(?:\.\d+)?)\s*(ml|milliliters?|l|liters?|cc)\b', 0.95),
            (r'\b(\d+(?:\.\d+)?)\s*(units?|iu|international units?)\b', 0.90),
            (r'\b(\d+(?:\.\d+)?)\s*(tablets?|pills?|capsules?|drops?)\b', 0.85),
            (r'\b(one|two|three|four|five|half|quarter)\s*(tablet|pill|capsule|drop)s?\b', 0.80),
        ]
        
        # Frequency patterns
        self.frequency_patterns = [
            (r'\b(once|twice|thrice|three times?)\s*(daily|a day|per day)\b', 0.95),
            (r'\b(daily|everyday|each day)\b', 0.95),
            (r'\b(bid|b\.i\.d\.?|twice daily)\b', 0.95),
            (r'\b(tid|t\.i\.d\.?|three times daily)\b', 0.95),
            (r'\b(qid|q\.i\.d\.?|four times daily)\b', 0.95),
            (r'\b(every \d+ hours?|q\d+h)\b', 0.90),
            (r'\b(morning|evening|night|bedtime)\b', 0.85),
            (r'\b(before meals?|after meals?|with meals?)\b', 0.85),
            (r'\b(as needed|prn|p\.r\.n\.?)\b', 0.80),
        ]
        
        # Route patterns
        self.route_patterns = [
            (r'\b(oral|orally|by mouth|po|p\.o\.?)\b', 0.90),
            (r'\b(topical|topically|apply)\b', 0.90),
            (r'\b(intravenous|iv|i\.v\.?)\b', 0.95),
            (r'\b(intramuscular|im|i\.m\.?)\b', 0.95),
            (r'\b(subcutaneous|sc|s\.c\.?|subq)\b', 0.95),
            (r'\b(inhaled?|inhalation)\b', 0.85),
        ]
        
        # Duration patterns
        self.duration_patterns = [
            (r'\bfor (\d+) (days?|weeks?|months?)\b', 0.90),
            (r'\b(\d+) (days?|weeks?|months?) course\b', 0.90),
            (r'\bcontinue for (\d+) (days?|weeks?|months?)\b', 0.85),
            (r'\buntil (symptoms improve|infection clears?|pain subsides?)\b', 0.75),
        ]
        
        # Condition patterns
        self.condition_patterns = [
            (r'\bfor (pain|fever|infection|inflammation|diabetes|hypertension)\b', 0.85),
            (r'\bto treat (pain|fever|infection|inflammation|diabetes|hypertension)\b', 0.85),
            (r'\b(headache|migraine|arthritis|asthma|copd|pneumonia)\b', 0.80),
            (r'\b(depression|anxiety|insomnia|nausea|vomiting)\b', 0.80),
        ]
    
    def process_prescription(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        validate_ontologies: bool = True,
        enhance_confidence: bool = True
    ) -> Dict[str, Any]:
        """Process a prescription text and extract entities."""
        start_time = time.time()
        
        try:
            # Extract entities
            entities = self._extract_entities(text)
            
            # Calculate overall confidence
            confidence_score = self._calculate_confidence(entities, text)
            
            # Generate warnings
            warnings = self._generate_warnings(entities, text)
            
            processing_time = time.time() - start_time
            
            parsed_prescription = {
                "original_text": text,
                "entities": [self._entity_to_dict(e) for e in entities],
                "confidence_score": confidence_score,
                "processing_time": processing_time,
                "warnings": warnings,
                "metadata": metadata or {}
            }
            
            return {
                "success": True,
                "parsed_prescription": parsed_prescription,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Prescription processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract medical entities from text using pattern matching."""
        entities = []
        text_lower = text.lower()
        
        # Extract drugs
        for pattern, confidence in self.drug_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(MedicalEntity(
                    entity_type="DRUG",
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence,
                    normalized_form=match.group().lower()
                ))
        
        # Extract dosages
        for pattern, confidence in self.dosage_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(MedicalEntity(
                    entity_type="DOSAGE",
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))
        
        # Extract frequencies
        for pattern, confidence in self.frequency_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(MedicalEntity(
                    entity_type="FREQUENCY",
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))
        
        # Extract routes
        for pattern, confidence in self.route_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(MedicalEntity(
                    entity_type="ROUTE",
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))
        
        # Extract durations
        for pattern, confidence in self.duration_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(MedicalEntity(
                    entity_type="DURATION",
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))
        
        # Extract conditions
        for pattern, confidence in self.condition_patterns:
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(MedicalEntity(
                    entity_type="CONDITION",
                    text=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=confidence
                ))
        
        # Remove duplicates and overlapping entities
        entities = self._remove_overlapping_entities(entities)
        
        # Filter by confidence threshold
        entities = [e for e in entities if e.confidence >= self.confidence_threshold]
        
        return entities
    
    def _remove_overlapping_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        """Remove overlapping entities, keeping the highest confidence ones."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_pos)
        
        filtered = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in filtered:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Overlapping - keep the higher confidence one
                    if entity.confidence > existing.confidence:
                        filtered.remove(existing)
                    else:
                        overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    def _calculate_confidence(self, entities: List[MedicalEntity], text: str) -> float:
        """Calculate overall confidence score."""
        if not entities:
            return 0.3  # Low confidence for no entities found
        
        # Average entity confidence
        avg_confidence = sum(e.confidence for e in entities) / len(entities)
        
        # Boost confidence based on entity types found
        entity_types = set(e.entity_type for e in entities)
        type_bonus = 0.0
        
        if "DRUG" in entity_types:
            type_bonus += 0.1
        if "DOSAGE" in entity_types:
            type_bonus += 0.1
        if "FREQUENCY" in entity_types:
            type_bonus += 0.05
        
        final_confidence = min(0.95, avg_confidence + type_bonus)
        return round(final_confidence, 2)
    
    def _generate_warnings(self, entities: List[MedicalEntity], text: str) -> List[str]:
        """Generate warnings based on extracted entities."""
        warnings = []
        
        if not entities:
            warnings.append("No medical entities detected in the prescription text.")
        
        # Check for missing critical information
        entity_types = set(e.entity_type for e in entities)
        
        if "DRUG" in entity_types and "DOSAGE" not in entity_types:
            warnings.append("Drug mentioned without clear dosage information.")
        
        if "DRUG" in entity_types and "FREQUENCY" not in entity_types:
            warnings.append("Drug mentioned without frequency information.")
        
        # Check for potential issues
        drug_entities = [e for e in entities if e.entity_type == "DRUG"]
        if len(drug_entities) > 5:
            warnings.append("Multiple drugs detected - verify for potential interactions.")
        
        return warnings
    
    def _entity_to_dict(self, entity: MedicalEntity) -> Dict[str, Any]:
        """Convert entity to dictionary format."""
        return {
            "entity_type": entity.entity_type,
            "text": entity.text,
            "start_pos": entity.start_pos,
            "end_pos": entity.end_pos,
            "confidence": entity.confidence,
            "normalized_form": entity.normalized_form
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get supported input formats."""
        return ["free_text", "structured", "clinical_notes"]
    
    def get_service_info(self) -> Dict[str, Any]:
        """Get service information."""
        return {
            "service_name": "OptimizedPrescriptionService",
            "version": "1.0.0",
            "supported_entities": ["DRUG", "DOSAGE", "FREQUENCY", "ROUTE", "DURATION", "CONDITION"],
            "confidence_threshold": self.confidence_threshold,
            "processing_method": "rule_based_ner"
        }
    
    def batch_process_prescriptions(
        self,
        prescriptions: List[Dict[str, Any]],
        validate_ontologies: bool = True,
        enhance_confidence: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple prescriptions in batch."""
        results = []
        
        for i, prescription in enumerate(prescriptions):
            text = prescription.get('text', '')
            metadata = prescription.get('metadata', {})
            metadata['batch_index'] = i
            
            result = self.process_prescription(
                text=text,
                metadata=metadata,
                validate_ontologies=validate_ontologies,
                enhance_confidence=enhance_confidence
            )
            results.append(result)
        
        return results
    
    def get_processing_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate processing statistics for batch results."""
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        
        total_entities = 0
        total_time = 0.0
        
        for result in results:
            if result.get('success'):
                parsed = result.get('parsed_prescription', {})
                total_entities += len(parsed.get('entities', []))
                total_time += parsed.get('processing_time', 0)
        
        return {
            "total_prescriptions": total,
            "successful_processing": successful,
            "error_rate": (total - successful) / total if total > 0 else 0,
            "average_entities_per_prescription": total_entities / successful if successful > 0 else 0,
            "average_processing_time": total_time / successful if successful > 0 else 0,
            "total_processing_time": total_time
        }

# Global instance
_global_prescription_service = None

def get_prescription_service() -> OptimizedPrescriptionService:
    """Get or create global prescription service instance."""
    global _global_prescription_service
    if _global_prescription_service is None:
        _global_prescription_service = OptimizedPrescriptionService()
    return _global_prescription_service