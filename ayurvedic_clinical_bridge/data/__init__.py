"""
Data processing and input handling for prescription parsing and validation.
"""

from .prescription_input import (
    PrescriptionInput,
    InputFormat,
    ValidationStatus,
    OntologyValidation,
    InputFormatDetector,
    MedicalOntologyValidator,
    ConfidenceScorer,
    PrescriptionInputProcessor
)

from .dataset_integration import (
    AyurvedicDatasetIntegrator,
    DatasetMetadata,
    ProcessedEntity,
    ValidationResult
)

__all__ = [
    'PrescriptionInput',
    'InputFormat',
    'ValidationStatus',
    'OntologyValidation',
    'InputFormatDetector',
    'MedicalOntologyValidator',
    'ConfidenceScorer',
    'PrescriptionInputProcessor',
    'AyurvedicDatasetIntegrator',
    'DatasetMetadata',
    'ProcessedEntity',
    'ValidationResult'
]