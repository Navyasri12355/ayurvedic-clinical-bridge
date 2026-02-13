
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class MedicalEntity:
    """
    Represents an extracted medical entity.
    """
    id: str
    text: str
    label: str  # NER label (e.g., 'DISEASE', 'HERB', 'SYMPTOM')
    start_char: int
    end_char: int
    confidence: float = 1.0
    normalized_text: Optional[str] = None
    ontology_id: Optional[str] = None
    
    # Validation fields
    is_validated: bool = False
    validation_source: Optional[str] = None
    ontology_codes: Dict[str, str] = field(default_factory=dict)
    
    @property
    def name(self) -> str:
        """Alias for text for backward compatibility."""
        return self.text
        
    @property
    def category(self) -> str:
        """Alias for label for backward compatibility."""
        return self.label

@dataclass
class ParsedPrescription:
    """
    Represents a parsed prescription with extracted entities.
    """
    original_text: str
    entities: List[MedicalEntity] = field(default_factory=list)
    diseases: List[MedicalEntity] = field(default_factory=list)
    medications: List[MedicalEntity] = field(default_factory=list)
    dosages: List[MedicalEntity] = field(default_factory=list)
    treatment_intent: List[MedicalEntity] = field(default_factory=list)
    
    # Metadata
    patient_info: Dict[str, Any] = field(default_factory=dict)
    doctor_info: Dict[str, Any] = field(default_factory=dict)
    date: Optional[str] = None
    confidence_score: float = 0.0
