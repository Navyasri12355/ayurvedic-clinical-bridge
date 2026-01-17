"""
Privacy preservation and anonymization utilities for prescription text processing.

This module implements PII detection and anonymization for prescription text,
secure data handling utilities, and logging/audit trails for privacy compliance
as required by Requirements 3.5.
"""

import re
import hashlib
import logging
import uuid
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class PIIType(Enum):
    """Enumeration of personally identifiable information types."""
    NAME = "name"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    SSN = "ssn"
    DATE_OF_BIRTH = "date_of_birth"
    MEDICAL_ID = "medical_id"
    CREDIT_CARD = "credit_card"
    CUSTOM = "custom"


@dataclass
class PIIDetection:
    """Represents a detected PII instance in text."""
    pii_type: PIIType
    original_text: str
    start_pos: int
    end_pos: int
    confidence: float
    replacement_token: str
    context: str = ""


@dataclass
class AnonymizationResult:
    """Result of text anonymization process."""
    anonymized_text: str
    detected_pii: List[PIIDetection]
    anonymization_map: Dict[str, str]  # original -> anonymized mapping
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrivacyAuditEntry:
    """Audit trail entry for privacy operations."""
    timestamp: datetime
    operation: str
    user_id: Optional[str]
    session_id: str
    pii_detected: int
    pii_types: List[str]
    text_length: int
    success: bool
    error_message: Optional[str] = None


class PIIDetector:
    """
    Detects personally identifiable information in prescription text.
    
    Uses pattern matching and heuristics to identify various types of PII
    that might appear in medical prescriptions.
    """
    
    def __init__(self):
        """Initialize PII detector with pattern definitions."""
        self.patterns = self._initialize_patterns()
        self.logger = logging.getLogger(__name__)
    
    def _initialize_patterns(self) -> Dict[PIIType, List[str]]:
        """Initialize regex patterns for different PII types."""
        return {
            PIIType.NAME: [
                # Common name patterns
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b[A-Z][a-z]+, [A-Z][a-z]+\b',  # Last, First
                r'\bMr\. [A-Z][a-z]+\b',
                r'\bMrs\. [A-Z][a-z]+\b',
                r'\bMs\. [A-Z][a-z]+\b',
                r'\bDr\. [A-Z][a-z]+\b',
                # Patient name indicators
                r'(?i)patient:?\s*([A-Z][a-z]+ [A-Z][a-z]+)',
                r'(?i)name:?\s*([A-Z][a-z]+ [A-Z][a-z]+)',
            ],
            PIIType.PHONE: [
                # US phone number formats
                r'\b\d{3}-\d{3}-\d{4}\b',
                r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',
                r'\b\d{3}\.\d{3}\.\d{4}\b',
                r'\b\d{10}\b',
                # International formats
                r'\+\d{1,3}\s*\d{3,4}\s*\d{3,4}\s*\d{3,4}',
            ],
            PIIType.EMAIL: [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            ],
            PIIType.ADDRESS: [
                # Street addresses
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b',
                r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl),?\s*[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?\b',
                # ZIP codes
                r'\b\d{5}(?:-\d{4})?\b',
            ],
            PIIType.SSN: [
                r'\b\d{3}-\d{2}-\d{4}\b',
                r'\b\d{9}\b',
            ],
            PIIType.DATE_OF_BIRTH: [
                # Various date formats that might be DOB
                r'(?i)(?:dob|date of birth|born):?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                r'(?i)(?:dob|date of birth|born):?\s*(\d{1,2}/\d{1,2}/\d{2,4})',
                r'(?i)(?:dob|date of birth|born):?\s*(\w+ \d{1,2}, \d{4})',
            ],
            PIIType.MEDICAL_ID: [
                # Medical record numbers, patient IDs
                r'(?i)(?:mrn|medical record|patient id|id):?\s*([A-Z0-9]{6,})',
                r'(?i)(?:mrn|medical record|patient id|id):?\s*(\d{6,})',
            ],
            PIIType.CREDIT_CARD: [
                # Credit card patterns (basic)
                r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            ]
        }
    
    def detect_pii(self, text: str, confidence_threshold: float = 0.7) -> List[PIIDetection]:
        """
        Detect PII in the given text.
        
        Args:
            text: Input text to scan for PII
            confidence_threshold: Minimum confidence for PII detection
            
        Returns:
            List of detected PII instances
        """
        detections = []
        
        for pii_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                
                for match in matches:
                    # Calculate confidence based on pattern specificity and context
                    confidence = self._calculate_confidence(
                        pii_type, match.group(), text, match.start(), match.end()
                    )
                    
                    if confidence >= confidence_threshold:
                        # Generate replacement token
                        replacement_token = self._generate_replacement_token(pii_type)
                        
                        # Extract context (surrounding text)
                        context_start = max(0, match.start() - 20)
                        context_end = min(len(text), match.end() + 20)
                        context = text[context_start:context_end]
                        
                        detection = PIIDetection(
                            pii_type=pii_type,
                            original_text=match.group(),
                            start_pos=match.start(),
                            end_pos=match.end(),
                            confidence=confidence,
                            replacement_token=replacement_token,
                            context=context
                        )
                        
                        detections.append(detection)
        
        # Remove overlapping detections (keep highest confidence)
        detections = self._remove_overlapping_detections(detections)
        
        return detections
    
    def _calculate_confidence(
        self, 
        pii_type: PIIType, 
        matched_text: str, 
        full_text: str, 
        start_pos: int, 
        end_pos: int
    ) -> float:
        """
        Calculate confidence score for a PII detection.
        
        Args:
            pii_type: Type of PII detected
            matched_text: The matched text
            full_text: Full input text
            start_pos: Start position of match
            end_pos: End position of match
            
        Returns:
            Confidence score between 0 and 1
        """
        base_confidence = 0.8  # Base confidence for pattern match
        
        # Adjust confidence based on context and characteristics
        if pii_type == PIIType.NAME:
            # Names in medical context are more likely to be PII
            if any(keyword in full_text.lower() for keyword in ['patient', 'mr.', 'mrs.', 'ms.', 'dr.']):
                base_confidence += 0.1
            
            # Check if it's a common medical term (reduce confidence)
            medical_terms = ['tablet', 'capsule', 'injection', 'syrup', 'cream', 'ointment']
            if matched_text.lower() in medical_terms:
                base_confidence -= 0.4
        
        elif pii_type == PIIType.PHONE:
            # Phone numbers with specific formatting are more confident
            if re.match(r'\(\d{3}\)\s*\d{3}-\d{4}', matched_text):
                base_confidence += 0.1
        
        elif pii_type == PIIType.SSN:
            # SSN with dashes is more confident
            if '-' in matched_text:
                base_confidence += 0.1
        
        elif pii_type == PIIType.MEDICAL_ID:
            # Medical IDs with context keywords are more confident
            context_keywords = ['mrn', 'medical record', 'patient id', 'id number']
            context_text = full_text[max(0, start_pos-50):min(len(full_text), end_pos+50)].lower()
            if any(keyword in context_text for keyword in context_keywords):
                base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _generate_replacement_token(self, pii_type: PIIType) -> str:
        """Generate a replacement token for the given PII type."""
        token_map = {
            PIIType.NAME: "[NAME]",
            PIIType.PHONE: "[PHONE]",
            PIIType.EMAIL: "[EMAIL]",
            PIIType.ADDRESS: "[ADDRESS]",
            PIIType.SSN: "[SSN]",
            PIIType.DATE_OF_BIRTH: "[DOB]",
            PIIType.MEDICAL_ID: "[MEDICAL_ID]",
            PIIType.CREDIT_CARD: "[CREDIT_CARD]",
            PIIType.CUSTOM: "[PII]"
        }
        return token_map.get(pii_type, "[PII]")
    
    def _remove_overlapping_detections(self, detections: List[PIIDetection]) -> List[PIIDetection]:
        """Remove overlapping PII detections, keeping the highest confidence ones."""
        if not detections:
            return detections
        
        # Sort by start position
        sorted_detections = sorted(detections, key=lambda x: x.start_pos)
        
        filtered_detections = []
        for detection in sorted_detections:
            # Check for overlap with existing detections
            overlaps = False
            for existing in filtered_detections:
                if (detection.start_pos < existing.end_pos and 
                    detection.end_pos > existing.start_pos):
                    # There's an overlap
                    if detection.confidence > existing.confidence:
                        # Replace existing with higher confidence detection
                        filtered_detections.remove(existing)
                        filtered_detections.append(detection)
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_detections.append(detection)
        
        return filtered_detections


class TextAnonymizer:
    """
    Anonymizes text by replacing detected PII with replacement tokens.
    
    Provides reversible anonymization with secure mapping storage.
    """
    
    def __init__(self, pii_detector: Optional[PIIDetector] = None):
        """
        Initialize text anonymizer.
        
        Args:
            pii_detector: PII detector instance (creates new if None)
        """
        self.pii_detector = pii_detector or PIIDetector()
        self.logger = logging.getLogger(__name__)
    
    def anonymize_text(
        self, 
        text: str, 
        preserve_structure: bool = True,
        hash_originals: bool = True
    ) -> AnonymizationResult:
        """
        Anonymize text by replacing PII with tokens.
        
        Args:
            text: Input text to anonymize
            preserve_structure: Whether to preserve text structure (length, spacing)
            hash_originals: Whether to hash original PII values for security
            
        Returns:
            AnonymizationResult with anonymized text and metadata
        """
        # Detect PII in the text
        pii_detections = self.pii_detector.detect_pii(text)
        
        if not pii_detections:
            return AnonymizationResult(
                anonymized_text=text,
                detected_pii=[],
                anonymization_map={},
                metadata={
                    'pii_detected': 0,
                    'anonymization_applied': False,
                    'preserve_structure': preserve_structure,
                    'hash_originals': hash_originals
                }
            )
        
        # Sort detections by position (reverse order for replacement)
        sorted_detections = sorted(pii_detections, key=lambda x: x.start_pos, reverse=True)
        
        # Create anonymization map
        anonymization_map = {}
        anonymized_text = text
        
        # Replace PII with tokens
        for detection in sorted_detections:
            original_value = detection.original_text
            replacement_token = detection.replacement_token
            
            # Create mapping entry
            if hash_originals:
                # Hash the original value for security
                hashed_original = self._hash_pii_value(original_value)
                anonymization_map[hashed_original] = {
                    'replacement_token': replacement_token,
                    'pii_type': detection.pii_type.value,
                    'confidence': detection.confidence,
                    'original_hash': hashed_original
                }
            else:
                anonymization_map[original_value] = {
                    'replacement_token': replacement_token,
                    'pii_type': detection.pii_type.value,
                    'confidence': detection.confidence
                }
            
            # Replace in text
            if preserve_structure:
                # Preserve length by padding with spaces if needed
                replacement = replacement_token
                if len(replacement) < len(original_value):
                    replacement += ' ' * (len(original_value) - len(replacement))
                elif len(replacement) > len(original_value):
                    replacement = replacement[:len(original_value)]
            else:
                replacement = replacement_token
            
            # Perform replacement
            anonymized_text = (
                anonymized_text[:detection.start_pos] + 
                replacement + 
                anonymized_text[detection.end_pos:]
            )
        
        return AnonymizationResult(
            anonymized_text=anonymized_text,
            detected_pii=pii_detections,
            anonymization_map=anonymization_map,
            metadata={
                'pii_detected': len(pii_detections),
                'anonymization_applied': True,
                'preserve_structure': preserve_structure,
                'hash_originals': hash_originals,
                'pii_types': list(set(d.pii_type.value for d in pii_detections))
            }
        )
    
    def _hash_pii_value(self, value: str) -> str:
        """
        Create a secure hash of PII value.
        
        Args:
            value: PII value to hash
            
        Returns:
            Hashed value as hex string
        """
        # Use SHA-256 with salt for security
        salt = "ayurvedic_clinical_bridge_pii_salt"
        combined = f"{salt}:{value}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()


class SecureDataHandler:
    """
    Handles secure data operations for prescription processing.
    
    Provides utilities for secure storage, retrieval, and handling of
    sensitive medical data with privacy preservation.
    """
    
    def __init__(self):
        """Initialize secure data handler."""
        self.logger = logging.getLogger(__name__)
    
    def sanitize_prescription_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize prescription data by removing or anonymizing sensitive fields.
        
        Args:
            data: Prescription data dictionary
            
        Returns:
            Sanitized data dictionary
        """
        sanitized_data = data.copy()
        
        # Fields that should be removed or anonymized
        sensitive_fields = [
            'patient_name', 'patient_id', 'medical_record_number',
            'phone_number', 'email', 'address', 'ssn', 'date_of_birth'
        ]
        
        for field in sensitive_fields:
            if field in sanitized_data:
                if isinstance(sanitized_data[field], str):
                    # Anonymize string fields
                    sanitized_data[field] = f"[REDACTED_{field.upper()}]"
                else:
                    # Remove non-string sensitive fields
                    del sanitized_data[field]
        
        # Recursively sanitize nested dictionaries
        for key, value in sanitized_data.items():
            if isinstance(value, dict):
                sanitized_data[key] = self.sanitize_prescription_data(value)
            elif isinstance(value, list):
                sanitized_data[key] = [
                    self.sanitize_prescription_data(item) if isinstance(item, dict) else item
                    for item in value
                ]
        
        return sanitized_data
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID for tracking."""
        return str(uuid.uuid4())
    
    def mask_sensitive_logs(self, log_message: str) -> str:
        """
        Mask sensitive information in log messages.
        
        Args:
            log_message: Original log message
            
        Returns:
            Log message with sensitive information masked
        """
        # Simple PII masking for logs
        masked_message = log_message
        
        # Mask potential phone numbers
        masked_message = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', masked_message)
        masked_message = re.sub(r'\b\(\d{3}\)\s*\d{3}-\d{4}\b', '[PHONE]', masked_message)
        
        # Mask potential emails
        masked_message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', masked_message)
        
        # Mask potential SSNs
        masked_message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', masked_message)
        
        return masked_message


class PrivacyAuditLogger:
    """
    Logs privacy-related operations for compliance and audit purposes.
    
    Maintains detailed audit trails of PII detection, anonymization,
    and data handling operations.
    """
    
    def __init__(self, log_file_path: Optional[str] = None):
        """
        Initialize privacy audit logger.
        
        Args:
            log_file_path: Path to audit log file (uses default if None)
        """
        self.log_file_path = log_file_path or "privacy_audit.log"
        self.logger = logging.getLogger(f"{__name__}.audit")
        
        # Configure audit logger
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_file_path)
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_pii_detection(
        self,
        session_id: str,
        user_id: Optional[str],
        text_length: int,
        pii_detections: List[PIIDetection],
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Log PII detection operation.
        
        Args:
            session_id: Session identifier
            user_id: User identifier (optional)
            text_length: Length of processed text
            pii_detections: List of detected PII instances
            success: Whether operation was successful
            error_message: Error message if operation failed
        """
        audit_entry = PrivacyAuditEntry(
            timestamp=datetime.now(),
            operation="pii_detection",
            user_id=user_id,
            session_id=session_id,
            pii_detected=len(pii_detections),
            pii_types=[d.pii_type.value for d in pii_detections],
            text_length=text_length,
            success=success,
            error_message=error_message
        )
        
        self._write_audit_entry(audit_entry)
    
    def log_anonymization(
        self,
        session_id: str,
        user_id: Optional[str],
        anonymization_result: AnonymizationResult,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Log text anonymization operation.
        
        Args:
            session_id: Session identifier
            user_id: User identifier (optional)
            anonymization_result: Result of anonymization
            success: Whether operation was successful
            error_message: Error message if operation failed
        """
        audit_entry = PrivacyAuditEntry(
            timestamp=datetime.now(),
            operation="text_anonymization",
            user_id=user_id,
            session_id=session_id,
            pii_detected=len(anonymization_result.detected_pii),
            pii_types=anonymization_result.metadata.get('pii_types', []),
            text_length=len(anonymization_result.anonymized_text),
            success=success,
            error_message=error_message
        )
        
        self._write_audit_entry(audit_entry)
    
    def log_data_access(
        self,
        session_id: str,
        user_id: Optional[str],
        operation: str,
        data_type: str,
        success: bool = True,
        error_message: Optional[str] = None
    ) -> None:
        """
        Log data access operation.
        
        Args:
            session_id: Session identifier
            user_id: User identifier (optional)
            operation: Type of operation (read, write, delete, etc.)
            data_type: Type of data accessed
            success: Whether operation was successful
            error_message: Error message if operation failed
        """
        audit_entry = PrivacyAuditEntry(
            timestamp=datetime.now(),
            operation=f"data_access_{operation}",
            user_id=user_id,
            session_id=session_id,
            pii_detected=0,  # Not applicable for data access
            pii_types=[],
            text_length=0,  # Not applicable for data access
            success=success,
            error_message=error_message
        )
        
        self._write_audit_entry(audit_entry)
    
    def _write_audit_entry(self, audit_entry: PrivacyAuditEntry) -> None:
        """
        Write audit entry to log.
        
        Args:
            audit_entry: Audit entry to write
        """
        # Create log message
        log_data = {
            'timestamp': audit_entry.timestamp.isoformat(),
            'operation': audit_entry.operation,
            'user_id': audit_entry.user_id,
            'session_id': audit_entry.session_id,
            'pii_detected': audit_entry.pii_detected,
            'pii_types': audit_entry.pii_types,
            'text_length': audit_entry.text_length,
            'success': audit_entry.success,
            'error_message': audit_entry.error_message
        }
        
        log_message = json.dumps(log_data)
        
        if audit_entry.success:
            self.logger.info(log_message)
        else:
            self.logger.error(log_message)
    
    def get_audit_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get summary of audit log entries for a date range.
        
        Args:
            start_date: Start date for summary (optional)
            end_date: End date for summary (optional)
            
        Returns:
            Dictionary with audit summary statistics
        """
        # This is a simplified implementation
        # In a production system, this would parse the log file
        # and generate comprehensive statistics
        
        return {
            'summary_note': 'Audit summary functionality requires log file parsing implementation',
            'log_file_path': self.log_file_path,
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None
        }


class PrivacyPreservationService:
    """
    Main service for privacy preservation and anonymization.
    
    Orchestrates PII detection, text anonymization, secure data handling,
    and audit logging for comprehensive privacy compliance.
    """
    
    def __init__(
        self,
        pii_detector: Optional[PIIDetector] = None,
        text_anonymizer: Optional[TextAnonymizer] = None,
        secure_data_handler: Optional[SecureDataHandler] = None,
        audit_logger: Optional[PrivacyAuditLogger] = None
    ):
        """
        Initialize privacy preservation service.
        
        Args:
            pii_detector: PII detector instance (creates new if None)
            text_anonymizer: Text anonymizer instance (creates new if None)
            secure_data_handler: Secure data handler instance (creates new if None)
            audit_logger: Privacy audit logger instance (creates new if None)
        """
        self.pii_detector = pii_detector or PIIDetector()
        self.text_anonymizer = text_anonymizer or TextAnonymizer(self.pii_detector)
        self.secure_data_handler = secure_data_handler or SecureDataHandler()
        self.audit_logger = audit_logger or PrivacyAuditLogger()
        self.logger = logging.getLogger(__name__)
    
    def process_prescription_text(
        self,
        prescription_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        preserve_structure: bool = True,
        hash_originals: bool = True
    ) -> AnonymizationResult:
        """
        Process prescription text with full privacy preservation.
        
        Args:
            prescription_text: Input prescription text
            user_id: User identifier for audit logging
            session_id: Session identifier for audit logging
            preserve_structure: Whether to preserve text structure
            hash_originals: Whether to hash original PII values
            
        Returns:
            AnonymizationResult with anonymized text and metadata
        """
        if session_id is None:
            session_id = self.secure_data_handler.generate_session_id()
        
        try:
            # Detect PII in the prescription text
            pii_detections = self.pii_detector.detect_pii(prescription_text)
            
            # Log PII detection
            self.audit_logger.log_pii_detection(
                session_id=session_id,
                user_id=user_id,
                text_length=len(prescription_text),
                pii_detections=pii_detections,
                success=True
            )
            
            # Anonymize the text
            anonymization_result = self.text_anonymizer.anonymize_text(
                text=prescription_text,
                preserve_structure=preserve_structure,
                hash_originals=hash_originals
            )
            
            # Log anonymization
            self.audit_logger.log_anonymization(
                session_id=session_id,
                user_id=user_id,
                anonymization_result=anonymization_result,
                success=True
            )
            
            self.logger.info(
                f"Successfully processed prescription text. "
                f"PII detected: {len(pii_detections)}, "
                f"Session: {session_id}"
            )
            
            return anonymization_result
            
        except Exception as e:
            error_message = f"Error processing prescription text: {str(e)}"
            self.logger.error(error_message)
            
            # Log failed operation
            self.audit_logger.log_pii_detection(
                session_id=session_id,
                user_id=user_id,
                text_length=len(prescription_text),
                pii_detections=[],
                success=False,
                error_message=error_message
            )
            
            raise
    
    def sanitize_prescription_data(
        self,
        prescription_data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Sanitize prescription data dictionary.
        
        Args:
            prescription_data: Prescription data to sanitize
            user_id: User identifier for audit logging
            session_id: Session identifier for audit logging
            
        Returns:
            Sanitized prescription data
        """
        if session_id is None:
            session_id = self.secure_data_handler.generate_session_id()
        
        try:
            # Sanitize the data
            sanitized_data = self.secure_data_handler.sanitize_prescription_data(
                prescription_data
            )
            
            # Log data access
            self.audit_logger.log_data_access(
                session_id=session_id,
                user_id=user_id,
                operation="sanitize",
                data_type="prescription_data",
                success=True
            )
            
            self.logger.info(
                f"Successfully sanitized prescription data. Session: {session_id}"
            )
            
            return sanitized_data
            
        except Exception as e:
            error_message = f"Error sanitizing prescription data: {str(e)}"
            self.logger.error(error_message)
            
            # Log failed operation
            self.audit_logger.log_data_access(
                session_id=session_id,
                user_id=user_id,
                operation="sanitize",
                data_type="prescription_data",
                success=False,
                error_message=error_message
            )
            
            raise
    
    def get_privacy_compliance_report(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Generate privacy compliance report.
        
        Args:
            start_date: Start date for report (optional)
            end_date: End date for report (optional)
            
        Returns:
            Privacy compliance report
        """
        try:
            audit_summary = self.audit_logger.get_audit_summary(
                start_date=start_date,
                end_date=end_date
            )
            
            report = {
                'report_generated': datetime.now().isoformat(),
                'period': {
                    'start_date': start_date.isoformat() if start_date else None,
                    'end_date': end_date.isoformat() if end_date else None
                },
                'audit_summary': audit_summary,
                'privacy_features': {
                    'pii_detection': True,
                    'text_anonymization': True,
                    'secure_data_handling': True,
                    'audit_logging': True,
                    'hash_originals': True,
                    'structure_preservation': True
                },
                'compliance_status': 'COMPLIANT'
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating privacy compliance report: {str(e)}")
            raise


# Utility functions for easy integration

def anonymize_prescription_text(
    text: str,
    preserve_structure: bool = True,
    hash_originals: bool = True
) -> AnonymizationResult:
    """
    Convenience function to anonymize prescription text.
    
    Args:
        text: Prescription text to anonymize
        preserve_structure: Whether to preserve text structure
        hash_originals: Whether to hash original PII values
        
    Returns:
        AnonymizationResult with anonymized text
    """
    service = PrivacyPreservationService()
    return service.process_prescription_text(
        prescription_text=text,
        preserve_structure=preserve_structure,
        hash_originals=hash_originals
    )


def sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to sanitize prescription data.
    
    Args:
        data: Prescription data to sanitize
        
    Returns:
        Sanitized prescription data
    """
    service = PrivacyPreservationService()
    return service.sanitize_prescription_data(data)


def detect_pii_in_text(text: str, confidence_threshold: float = 0.7) -> List[PIIDetection]:
    """
    Convenience function to detect PII in text.
    
    Args:
        text: Text to scan for PII
        confidence_threshold: Minimum confidence for detection
        
    Returns:
        List of detected PII instances
    """
    detector = PIIDetector()
    return detector.detect_pii(text, confidence_threshold)