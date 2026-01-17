"""
Prescription input processing and validation module.

This module handles prescription input processing, format detection, medical ontology
validation, and confidence scoring for extracted entities.
"""

from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from pathlib import Path
import requests
from urllib.parse import quote
import time

from ..models.hybrid_ner import MedicalEntity, ParsedPrescription
from ..utils.privacy_handler import PrivacyPreservationService, AnonymizationResult


class InputFormat(Enum):
    """Enumeration of supported prescription input formats."""
    FREE_TEXT = "free_text"
    STRUCTURED = "structured"
    OCR = "ocr"
    JSON = "json"
    XML = "xml"


class ValidationStatus(Enum):
    """Enumeration of validation status values."""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"
    NOT_FOUND = "not_found"


@dataclass
class PrescriptionInput:
    """
    Represents prescription input with format detection and metadata.
    
    This class extends the basic prescription input to include format detection,
    validation status, and processing metadata.
    """
    text: str
    format: InputFormat = InputFormat.FREE_TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    validation_status: ValidationStatus = ValidationStatus.UNCERTAIN
    detected_language: Optional[str] = None
    processing_timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.processing_timestamp is None:
            import datetime
            self.processing_timestamp = datetime.datetime.now().isoformat()


@dataclass
class OntologyValidation:
    """Represents validation results against medical ontologies."""
    entity: MedicalEntity
    icd10_codes: List[str] = field(default_factory=list)
    snomed_codes: List[str] = field(default_factory=list)
    validation_status: ValidationStatus = ValidationStatus.UNCERTAIN
    confidence_score: float = 0.0
    validation_source: str = ""
    alternative_terms: List[str] = field(default_factory=list)


class InputFormatDetector:
    """
    Detects the format of prescription input text.
    
    Analyzes input text to determine if it's free text, structured data,
    OCR output, JSON, or XML format.
    """
    
    def __init__(self):
        """Initialize the format detector with pattern definitions."""
        # Patterns for different input formats
        self.structured_patterns = [
            r'(?i)patient\s*:\s*\w+',
            r'(?i)diagnosis\s*:\s*\w+',
            r'(?i)medication\s*:\s*\w+',
            r'(?i)dosage\s*:\s*\w+',
            r'(?i)rx\s*:\s*\w+',
            r'(?i)sig\s*:\s*\w+',
        ]
        
        self.ocr_patterns = [
            r'[Il1|]{2,}',  # Common OCR misreads
            r'[0O]{2,}',    # Zero/O confusion
            r'\b[a-zA-Z]{1,2}\d+[a-zA-Z]*\b',  # Fragmented words
            r'[^\w\s]{3,}',  # Multiple special characters
        ]
        
        self.json_patterns = [
            r'^\s*\{.*\}\s*$',
            r'^\s*\[.*\]\s*$',
        ]
        
        self.xml_patterns = [
            r'<\?xml.*\?>',
            r'<[^>]+>.*</[^>]+>',
        ]
    
    def detect_format(self, text: str) -> InputFormat:
        """
        Detect the format of input text.
        
        Args:
            text: Input prescription text
            
        Returns:
            Detected InputFormat
        """
        if not text or not text.strip():
            return InputFormat.FREE_TEXT
        
        text = text.strip()
        
        # Check for JSON format
        if self._is_json_format(text):
            return InputFormat.JSON
        
        # Check for XML format
        if self._is_xml_format(text):
            return InputFormat.XML
        
        # Check for structured format
        if self._is_structured_format(text):
            return InputFormat.STRUCTURED
        
        # Check for OCR format (likely contains OCR artifacts)
        if self._is_ocr_format(text):
            return InputFormat.OCR
        
        # Default to free text
        return InputFormat.FREE_TEXT
    
    def _is_json_format(self, text: str) -> bool:
        """Check if text is in JSON format."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return any(re.search(pattern, text, re.DOTALL) for pattern in self.json_patterns)
    
    def _is_xml_format(self, text: str) -> bool:
        """Check if text is in XML format."""
        return any(re.search(pattern, text, re.DOTALL) for pattern in self.xml_patterns)
    
    def _is_structured_format(self, text: str) -> bool:
        """Check if text follows structured prescription format."""
        matches = sum(1 for pattern in self.structured_patterns 
                     if re.search(pattern, text))
        return matches >= 2  # At least 2 structured elements
    
    def _is_ocr_format(self, text: str) -> bool:
        """Check if text likely contains OCR artifacts."""
        ocr_score = sum(1 for pattern in self.ocr_patterns 
                       if re.search(pattern, text))
        return ocr_score >= 2  # Multiple OCR indicators


class MedicalOntologyValidator:
    """
    Validates medical entities against ICD-10 and SNOMED-CT ontologies.
    
    This class provides validation of extracted medical entities against
    standard medical ontologies to ensure accuracy and standardization.
    """
    
    def __init__(self, cache_size: int = 1000):
        """
        Initialize the ontology validator.
        
        Args:
            cache_size: Maximum number of validation results to cache
        """
        self.cache_size = cache_size
        self.validation_cache: Dict[str, OntologyValidation] = {}
        
        # ICD-10 API endpoints (using public APIs where available)
        self.icd10_api_base = "https://id.who.int/icd/release/11/2019-04"
        self.snomed_api_base = "https://browser.ihtsdotools.org/snowstorm/snomed-ct"
        
        # Load local ontology data if available
        self.local_icd10_codes = self._load_local_icd10_codes()
        self.local_snomed_codes = self._load_local_snomed_codes()
    
    def validate_entity(self, entity: MedicalEntity) -> OntologyValidation:
        """
        Validate a medical entity against ontologies.
        
        Args:
            entity: Medical entity to validate
            
        Returns:
            OntologyValidation result
        """
        # Check cache first
        cache_key = f"{entity.category}:{entity.name.lower()}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        validation = OntologyValidation(entity=entity)
        
        # Validate against ICD-10
        icd10_result = self._validate_icd10(entity)
        validation.icd10_codes = icd10_result.get('codes', [])
        
        # Validate against SNOMED-CT
        snomed_result = self._validate_snomed(entity)
        validation.snomed_codes = snomed_result.get('codes', [])
        
        # Determine overall validation status
        validation.validation_status = self._determine_validation_status(
            icd10_result, snomed_result
        )
        
        # Calculate confidence score
        validation.confidence_score = self._calculate_validation_confidence(
            icd10_result, snomed_result
        )
        
        # Set validation source
        validation.validation_source = self._get_validation_source(
            icd10_result, snomed_result
        )
        
        # Get alternative terms
        validation.alternative_terms = self._get_alternative_terms(
            icd10_result, snomed_result
        )
        
        # Cache the result
        if len(self.validation_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.validation_cache))
            del self.validation_cache[oldest_key]
        
        self.validation_cache[cache_key] = validation
        return validation
    
    def _validate_icd10(self, entity: MedicalEntity) -> Dict[str, Any]:
        """
        Validate entity against ICD-10 codes.
        
        Args:
            entity: Medical entity to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'codes': [],
            'confidence': 0.0,
            'source': 'local',
            'alternatives': []
        }
        
        # First check local ICD-10 codes
        local_matches = self._search_local_icd10(entity.name)
        if local_matches:
            result['codes'] = local_matches
            result['confidence'] = 0.8  # High confidence for exact matches
            result['source'] = 'local'
            return result
        
        # Try fuzzy matching with local codes
        fuzzy_matches = self._fuzzy_search_icd10(entity.name)
        if fuzzy_matches:
            result['codes'] = [match['code'] for match in fuzzy_matches[:3]]
            result['confidence'] = max(match['score'] for match in fuzzy_matches)
            result['alternatives'] = [match['term'] for match in fuzzy_matches]
            result['source'] = 'local_fuzzy'
            return result
        
        # If no local matches, try API (with rate limiting)
        try:
            api_result = self._query_icd10_api(entity.name)
            if api_result:
                result.update(api_result)
                result['source'] = 'api'
        except Exception as e:
            # API unavailable, continue with local results
            result['confidence'] = 0.1
            result['source'] = 'unavailable'
        
        return result
    
    def _validate_snomed(self, entity: MedicalEntity) -> Dict[str, Any]:
        """
        Validate entity against SNOMED-CT codes.
        
        Args:
            entity: Medical entity to validate
            
        Returns:
            Dictionary with validation results
        """
        result = {
            'codes': [],
            'confidence': 0.0,
            'source': 'local',
            'alternatives': []
        }
        
        # Check local SNOMED codes
        local_matches = self._search_local_snomed(entity.name)
        if local_matches:
            result['codes'] = local_matches
            result['confidence'] = 0.8
            result['source'] = 'local'
            return result
        
        # Try fuzzy matching
        fuzzy_matches = self._fuzzy_search_snomed(entity.name)
        if fuzzy_matches:
            result['codes'] = [match['code'] for match in fuzzy_matches[:3]]
            result['confidence'] = max(match['score'] for match in fuzzy_matches)
            result['alternatives'] = [match['term'] for match in fuzzy_matches]
            result['source'] = 'local_fuzzy'
        
        return result
    
    def _load_local_icd10_codes(self) -> Dict[str, str]:
        """Load local ICD-10 codes from file or return sample data."""
        # In a real implementation, this would load from a comprehensive database
        # For now, return a sample of common medical terms
        return {
            'hypertension': 'I10',
            'diabetes': 'E11',
            'diabetes mellitus': 'E11.9',
            'type 2 diabetes': 'E11',
            'high blood pressure': 'I10',
            'headache': 'R51',
            'fever': 'R50.9',
            'cough': 'R05',
            'chest pain': 'R07.9',
            'abdominal pain': 'R10.9',
            'nausea': 'R11',
            'vomiting': 'R11',
            'diarrhea': 'K59.1',
            'constipation': 'K59.0',
            'fatigue': 'R53',
            'insomnia': 'G47.0',
            'anxiety': 'F41.9',
            'depression': 'F32.9',
            'asthma': 'J45.9',
            'pneumonia': 'J18.9',
        }
    
    def _load_local_snomed_codes(self) -> Dict[str, str]:
        """Load local SNOMED-CT codes from file or return sample data."""
        # Sample SNOMED-CT codes for common medical terms
        return {
            'hypertension': '38341003',
            'diabetes': '73211009',
            'diabetes mellitus': '73211009',
            'type 2 diabetes': '44054006',
            'high blood pressure': '38341003',
            'headache': '25064002',
            'fever': '386661006',
            'cough': '49727002',
            'chest pain': '29857009',
            'abdominal pain': '21522001',
            'nausea': '422587007',
            'vomiting': '422400008',
            'diarrhea': '62315008',
            'constipation': '14760008',
            'fatigue': '84229001',
            'insomnia': '193462001',
            'anxiety': '48694002',
            'depression': '35489007',
            'asthma': '195967001',
            'pneumonia': '233604007',
        }
    
    def _search_local_icd10(self, term: str) -> List[str]:
        """Search for exact matches in local ICD-10 codes."""
        term_lower = term.lower().strip()
        matches = []
        
        for medical_term, code in self.local_icd10_codes.items():
            if term_lower == medical_term.lower():
                matches.append(code)
        
        return matches
    
    def _search_local_snomed(self, term: str) -> List[str]:
        """Search for exact matches in local SNOMED codes."""
        term_lower = term.lower().strip()
        matches = []
        
        for medical_term, code in self.local_snomed_codes.items():
            if term_lower == medical_term.lower():
                matches.append(code)
        
        return matches
    
    def _fuzzy_search_icd10(self, term: str) -> List[Dict[str, Any]]:
        """Perform fuzzy search on ICD-10 codes."""
        term_lower = term.lower().strip()
        matches = []
        
        for medical_term, code in self.local_icd10_codes.items():
            # Simple fuzzy matching based on substring and similarity
            score = self._calculate_similarity(term_lower, medical_term.lower())
            if score > 0.6:  # Threshold for fuzzy matching
                matches.append({
                    'code': code,
                    'term': medical_term,
                    'score': score
                })
        
        return sorted(matches, key=lambda x: x['score'], reverse=True)
    
    def _fuzzy_search_snomed(self, term: str) -> List[Dict[str, Any]]:
        """Perform fuzzy search on SNOMED codes."""
        term_lower = term.lower().strip()
        matches = []
        
        for medical_term, code in self.local_snomed_codes.items():
            score = self._calculate_similarity(term_lower, medical_term.lower())
            if score > 0.6:
                matches.append({
                    'code': code,
                    'term': medical_term,
                    'score': score
                })
        
        return sorted(matches, key=lambda x: x['score'], reverse=True)
    
    def _calculate_similarity(self, term1: str, term2: str) -> float:
        """
        Calculate similarity between two terms using simple string metrics.
        
        Args:
            term1: First term
            term2: Second term
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple Jaccard similarity for words
        words1 = set(term1.split())
        words2 = set(term2.split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Also consider substring matching
        substring_score = 0.0
        if term1 in term2 or term2 in term1:
            substring_score = 0.5
        
        # Combine scores
        return max(jaccard, substring_score)
    
    def _query_icd10_api(self, term: str) -> Optional[Dict[str, Any]]:
        """
        Query ICD-10 API for term validation.
        
        Note: This is a placeholder for actual API integration.
        Real implementation would use WHO ICD-11 API or similar service.
        """
        # Placeholder for API integration
        # In real implementation, this would make HTTP requests to ICD API
        return None
    
    def _determine_validation_status(
        self, 
        icd10_result: Dict[str, Any], 
        snomed_result: Dict[str, Any]
    ) -> ValidationStatus:
        """Determine overall validation status from ontology results."""
        icd10_confidence = icd10_result.get('confidence', 0.0)
        snomed_confidence = snomed_result.get('confidence', 0.0)
        
        max_confidence = max(icd10_confidence, snomed_confidence)
        
        if max_confidence >= 0.8:
            return ValidationStatus.VALID
        elif max_confidence >= 0.6:
            return ValidationStatus.UNCERTAIN
        elif max_confidence > 0.0:
            return ValidationStatus.UNCERTAIN
        else:
            return ValidationStatus.NOT_FOUND
    
    def _calculate_validation_confidence(
        self, 
        icd10_result: Dict[str, Any], 
        snomed_result: Dict[str, Any]
    ) -> float:
        """Calculate overall validation confidence score."""
        icd10_confidence = icd10_result.get('confidence', 0.0)
        snomed_confidence = snomed_result.get('confidence', 0.0)
        
        # Weight ICD-10 and SNOMED equally, take the maximum
        return max(icd10_confidence, snomed_confidence)
    
    def _get_validation_source(
        self, 
        icd10_result: Dict[str, Any], 
        snomed_result: Dict[str, Any]
    ) -> str:
        """Get the primary validation source."""
        icd10_confidence = icd10_result.get('confidence', 0.0)
        snomed_confidence = snomed_result.get('confidence', 0.0)
        
        if icd10_confidence >= snomed_confidence:
            return f"ICD-10 ({icd10_result.get('source', 'unknown')})"
        else:
            return f"SNOMED-CT ({snomed_result.get('source', 'unknown')})"
    
    def _get_alternative_terms(
        self, 
        icd10_result: Dict[str, Any], 
        snomed_result: Dict[str, Any]
    ) -> List[str]:
        """Get alternative terms from validation results."""
        alternatives = []
        alternatives.extend(icd10_result.get('alternatives', []))
        alternatives.extend(snomed_result.get('alternatives', []))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_alternatives = []
        for alt in alternatives:
            if alt.lower() not in seen:
                seen.add(alt.lower())
                unique_alternatives.append(alt)
        
        return unique_alternatives[:5]  # Limit to top 5 alternatives


class ConfidenceScorer:
    """
    Calculates confidence scores for extracted entities based on multiple factors.
    
    This class provides comprehensive confidence scoring that considers model
    confidence, ontology validation, context consistency, and other factors.
    """
    
    def __init__(self):
        """Initialize the confidence scorer."""
        self.weights = {
            'model_confidence': 0.4,
            'ontology_validation': 0.3,
            'context_consistency': 0.2,
            'entity_completeness': 0.1
        }
    
    def calculate_entity_confidence(
        self, 
        entity: MedicalEntity,
        validation: OntologyValidation,
        context_entities: List[MedicalEntity]
    ) -> float:
        """
        Calculate comprehensive confidence score for an entity.
        
        Args:
            entity: Medical entity to score
            validation: Ontology validation results
            context_entities: Other entities in the same prescription
            
        Returns:
            Confidence score between 0 and 1
        """
        # Model confidence (from NER model)
        model_score = entity.confidence
        
        # Ontology validation score
        ontology_score = validation.confidence_score
        
        # Context consistency score
        context_score = self._calculate_context_consistency(entity, context_entities)
        
        # Entity completeness score
        completeness_score = self._calculate_entity_completeness(entity)
        
        # Weighted combination
        total_confidence = (
            self.weights['model_confidence'] * model_score +
            self.weights['ontology_validation'] * ontology_score +
            self.weights['context_consistency'] * context_score +
            self.weights['entity_completeness'] * completeness_score
        )
        
        return min(1.0, max(0.0, total_confidence))
    
    def calculate_prescription_confidence(
        self, 
        parsed_prescription: ParsedPrescription,
        validations: List[OntologyValidation]
    ) -> Dict[str, float]:
        """
        Calculate confidence scores for the entire prescription.
        
        Args:
            parsed_prescription: Parsed prescription with entities
            validations: List of ontology validations
            
        Returns:
            Dictionary with confidence scores for different aspects
        """
        all_entities = (
            parsed_prescription.diseases +
            parsed_prescription.medications +
            parsed_prescription.dosages +
            parsed_prescription.treatment_intent
        )
        
        if not all_entities:
            return {
                'overall': 0.0,
                'diseases': 0.0,
                'medications': 0.0,
                'dosages': 0.0,
                'treatment_intent': 0.0,
                'completeness': 0.0,
                'consistency': 0.0
            }
        
        # Calculate individual entity confidences
        entity_confidences = []
        for entity in all_entities:
            # Find corresponding validation
            validation = next(
                (v for v in validations if v.entity.id == entity.id),
                OntologyValidation(entity=entity)
            )
            confidence = self.calculate_entity_confidence(entity, validation, all_entities)
            entity_confidences.append(confidence)
        
        # Category-specific confidences
        disease_confidences = [
            self.calculate_entity_confidence(
                entity, 
                next((v for v in validations if v.entity.id == entity.id), 
                     OntologyValidation(entity=entity)),
                all_entities
            )
            for entity in parsed_prescription.diseases
        ]
        
        medication_confidences = [
            self.calculate_entity_confidence(
                entity,
                next((v for v in validations if v.entity.id == entity.id),
                     OntologyValidation(entity=entity)),
                all_entities
            )
            for entity in parsed_prescription.medications
        ]
        
        dosage_confidences = [
            self.calculate_entity_confidence(
                entity,
                next((v for v in validations if v.entity.id == entity.id),
                     OntologyValidation(entity=entity)),
                all_entities
            )
            for entity in parsed_prescription.dosages
        ]
        
        treatment_confidences = [
            self.calculate_entity_confidence(
                entity,
                next((v for v in validations if v.entity.id == entity.id),
                     OntologyValidation(entity=entity)),
                all_entities
            )
            for entity in parsed_prescription.treatment_intent
        ]
        
        # Calculate completeness score
        completeness_score = self._calculate_prescription_completeness(parsed_prescription)
        
        # Calculate consistency score
        consistency_score = self._calculate_prescription_consistency(all_entities)
        
        return {
            'overall': sum(entity_confidences) / len(entity_confidences),
            'diseases': sum(disease_confidences) / len(disease_confidences) if disease_confidences else 0.0,
            'medications': sum(medication_confidences) / len(medication_confidences) if medication_confidences else 0.0,
            'dosages': sum(dosage_confidences) / len(dosage_confidences) if dosage_confidences else 0.0,
            'treatment_intent': sum(treatment_confidences) / len(treatment_confidences) if treatment_confidences else 0.0,
            'completeness': completeness_score,
            'consistency': consistency_score
        }
    
    def _calculate_context_consistency(
        self, 
        entity: MedicalEntity, 
        context_entities: List[MedicalEntity]
    ) -> float:
        """Calculate how consistent an entity is with its context."""
        if not context_entities:
            return 0.5  # Neutral score when no context
        
        # Simple heuristic: check if entity type is appropriate for context
        entity_types = [e.category for e in context_entities]
        
        # Expected combinations
        if entity.category == 'DISEASE':
            # Diseases should have medications and possibly dosages
            if 'DRUG' in entity_types:
                return 0.8
            else:
                return 0.4
        
        elif entity.category == 'DRUG':
            # Drugs should have diseases and possibly dosages
            if 'DISEASE' in entity_types:
                return 0.8
            else:
                return 0.6
        
        elif entity.category == 'DOSAGE':
            # Dosages should have drugs
            if 'DRUG' in entity_types:
                return 0.9
            else:
                return 0.3
        
        return 0.5  # Default neutral score
    
    def _calculate_entity_completeness(self, entity: MedicalEntity) -> float:
        """Calculate completeness score for an entity."""
        score = 0.5  # Base score
        
        # Check if entity has ontology codes
        if entity.ontology_codes:
            score += 0.2
        
        # Check if entity has synonyms
        if entity.synonyms:
            score += 0.1
        
        # Check entity name length (not too short or too long)
        name_length = len(entity.name.strip())
        if 3 <= name_length <= 50:
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_prescription_completeness(self, prescription: ParsedPrescription) -> float:
        """Calculate completeness score for the entire prescription."""
        score = 0.0
        
        # Check presence of key components
        if prescription.diseases:
            score += 0.3
        if prescription.medications:
            score += 0.4
        if prescription.dosages:
            score += 0.2
        if prescription.treatment_intent:
            score += 0.1
        
        return score
    
    def _calculate_prescription_consistency(self, entities: List[MedicalEntity]) -> float:
        """Calculate consistency score across all entities."""
        if len(entities) < 2:
            return 0.5  # Neutral for single entity
        
        # Check for logical relationships
        has_disease = any(e.category == 'DISEASE' for e in entities)
        has_drug = any(e.category == 'DRUG' for e in entities)
        has_dosage = any(e.category == 'DOSAGE' for e in entities)
        
        consistency_score = 0.0
        
        # Disease-drug consistency
        if has_disease and has_drug:
            consistency_score += 0.4
        
        # Drug-dosage consistency
        if has_drug and has_dosage:
            consistency_score += 0.3
        
        # Overall entity diversity
        unique_categories = len(set(e.category for e in entities))
        if unique_categories >= 2:
            consistency_score += 0.3
        
        return min(1.0, consistency_score)


class PrescriptionInputProcessor:
    """
    Main processor for prescription input handling, validation, and confidence scoring.
    
    This class orchestrates the entire prescription input processing pipeline,
    including format detection, ontology validation, confidence scoring, and privacy preservation.
    """
    
    def __init__(self, enable_privacy_preservation: bool = True):
        """
        Initialize the prescription input processor.
        
        Args:
            enable_privacy_preservation: Whether to enable privacy preservation features
        """
        self.format_detector = InputFormatDetector()
        self.ontology_validator = MedicalOntologyValidator()
        self.confidence_scorer = ConfidenceScorer()
        
        # Initialize privacy preservation service
        self.enable_privacy_preservation = enable_privacy_preservation
        if enable_privacy_preservation:
            self.privacy_service = PrivacyPreservationService()
        else:
            self.privacy_service = None
    
    def process_prescription_input(
        self, 
        text: str, 
        metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        preserve_structure: bool = True
    ) -> Dict[str, Any]:
        """
        Process raw prescription input text with privacy preservation.
        
        Args:
            text: Raw prescription text
            metadata: Optional metadata about the input
            user_id: User identifier for audit logging
            session_id: Session identifier for audit logging
            preserve_structure: Whether to preserve text structure during anonymization
            
        Returns:
            Dictionary containing processed input and anonymization results
        """
        if metadata is None:
            metadata = {}
        
        # Apply privacy preservation if enabled
        anonymization_result = None
        processed_text = text
        
        if self.enable_privacy_preservation and self.privacy_service:
            try:
                anonymization_result = self.privacy_service.process_prescription_text(
                    prescription_text=text,
                    user_id=user_id,
                    session_id=session_id,
                    preserve_structure=preserve_structure,
                    hash_originals=True
                )
                processed_text = anonymization_result.anonymized_text
                
                # Add privacy metadata
                metadata.update({
                    'privacy_preserved': True,
                    'pii_detected': len(anonymization_result.detected_pii),
                    'pii_types': anonymization_result.metadata.get('pii_types', []),
                    'anonymization_applied': anonymization_result.metadata.get('anonymization_applied', False)
                })
                
            except Exception as e:
                # Log error but continue processing with original text
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Privacy preservation failed: {str(e)}")
                metadata['privacy_error'] = str(e)
                metadata['privacy_preserved'] = False
        else:
            metadata['privacy_preserved'] = False
        
        # Detect input format using processed (anonymized) text
        detected_format = self.format_detector.detect_format(processed_text)
        
        # Create prescription input object
        prescription_input = PrescriptionInput(
            text=processed_text,  # Use anonymized text for further processing
            format=detected_format,
            metadata=metadata
        )
        
        # Add format-specific metadata
        prescription_input.metadata.update({
            'detected_format': detected_format.value,
            'text_length': len(processed_text),
            'word_count': len(processed_text.split()) if processed_text else 0,
            'has_special_chars': bool(re.search(r'[^\w\s]', processed_text)) if processed_text else False,
            'original_text_length': len(text),
            'privacy_preservation_enabled': self.enable_privacy_preservation
        })
        
        return {
            'prescription_input': prescription_input,
            'anonymization_result': anonymization_result,
            'original_text': text if not self.enable_privacy_preservation else None,  # Only include if privacy is disabled
            'processed_text': processed_text
        }
    
    def validate_parsed_prescription(
        self, 
        parsed_prescription: ParsedPrescription
    ) -> Dict[str, Any]:
        """
        Validate and enhance a parsed prescription with ontology validation and confidence scoring.
        
        Args:
            parsed_prescription: Parsed prescription from NER model
            
        Returns:
            Dictionary with validation results and enhanced confidence scores
        """
        all_entities = (
            parsed_prescription.diseases +
            parsed_prescription.medications +
            parsed_prescription.dosages +
            parsed_prescription.treatment_intent
        )
        
        # Validate each entity against ontologies
        validations = []
        for entity in all_entities:
            validation = self.ontology_validator.validate_entity(entity)
            validations.append(validation)
            
            # Update entity with ontology codes if found
            if validation.icd10_codes or validation.snomed_codes:
                entity.ontology_codes = validation.icd10_codes + validation.snomed_codes
        
        # Calculate enhanced confidence scores
        enhanced_confidence_scores = self.confidence_scorer.calculate_prescription_confidence(
            parsed_prescription, validations
        )
        
        # Update prescription confidence scores
        parsed_prescription.confidence_scores.update(enhanced_confidence_scores)
        
        # Prepare validation summary
        validation_summary = {
            'total_entities': len(all_entities),
            'validated_entities': len([v for v in validations if v.validation_status == ValidationStatus.VALID]),
            'uncertain_entities': len([v for v in validations if v.validation_status == ValidationStatus.UNCERTAIN]),
            'invalid_entities': len([v for v in validations if v.validation_status == ValidationStatus.INVALID]),
            'not_found_entities': len([v for v in validations if v.validation_status == ValidationStatus.NOT_FOUND]),
            'average_validation_confidence': sum(v.confidence_score for v in validations) / len(validations) if validations else 0.0,
            'ontology_coverage': {
                'icd10_matches': len([v for v in validations if v.icd10_codes]),
                'snomed_matches': len([v for v in validations if v.snomed_codes])
            }
        }
        
        return {
            'parsed_prescription': parsed_prescription,
            'validations': validations,
            'validation_summary': validation_summary,
            'enhanced_confidence_scores': enhanced_confidence_scores
        }
    
    def sanitize_prescription_data(
        self,
        prescription_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sanitize prescription data dictionary with privacy preservation.
        
        Args:
            prescription_data: Prescription data to sanitize
            user_id: User identifier for audit logging
            session_id: Session identifier for audit logging
            
        Returns:
            Sanitized prescription data
        """
        if not self.enable_privacy_preservation or not self.privacy_service:
            return prescription_data
        
        try:
            sanitized_data = self.privacy_service.sanitize_prescription_data(
                prescription_data=prescription_data,
                user_id=user_id,
                session_id=session_id
            )
            return sanitized_data
        except Exception as e:
            # Log error but return original data
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Data sanitization failed: {str(e)}")
            return prescription_data
    
    def get_privacy_compliance_report(
        self,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Generate privacy compliance report.
        
        Args:
            start_date: Start date for report (optional)
            end_date: End date for report (optional)
            
        Returns:
            Privacy compliance report or None if privacy is disabled
        """
        if not self.enable_privacy_preservation or not self.privacy_service:
            return None
        
        try:
            return self.privacy_service.get_privacy_compliance_report(
                start_date=start_date,
                end_date=end_date
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Privacy compliance report generation failed: {str(e)}")
            return None