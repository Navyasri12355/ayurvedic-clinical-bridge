"""
Request/response validation and sanitization middleware.

This module provides comprehensive validation capabilities including:
- Input sanitization and security validation
- Request/response schema validation
- Data type validation and conversion
- Security headers and CORS validation
- Content validation and filtering

Requirements: All requirements - system reliability and security
"""

import re
import json
import html
from typing import Dict, List, Optional, Any, Union, Type
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from pydantic import BaseModel, ValidationError, Field, validator
from loguru import logger


class ValidationSeverity(Enum):
    """Validation error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Validation result container."""
    is_valid: bool
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    sanitized_data: Optional[Any] = None
    metadata: Dict[str, Any] = None


class SecurityValidator:
    """Security-focused validation utilities."""
    
    # Common injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)",
        r"(--|#|/\*|\*/)",
        r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
        r"(\bUNION\s+SELECT\b)",
        r"(\b(EXEC|EXECUTE)\s*\()",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>.*?</iframe>",
        r"<object[^>]*>.*?</object>",
        r"<embed[^>]*>.*?</embed>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$(){}[\]\\]",
        r"\b(rm|del|format|fdisk|mkfs)\b",
        r"\b(wget|curl|nc|netcat)\b",
        r"\b(eval|exec|system)\b",
    ]
    
    @classmethod
    def detect_sql_injection(cls, text: str) -> List[str]:
        """Detect potential SQL injection attempts."""
        detected = []
        text_upper = text.upper()
        
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, text_upper, re.IGNORECASE):
                detected.append(f"SQL injection pattern detected: {pattern}")
        
        return detected
    
    @classmethod
    def detect_xss(cls, text: str) -> List[str]:
        """Detect potential XSS attempts."""
        detected = []
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"XSS pattern detected: {pattern}")
        
        return detected
    
    @classmethod
    def detect_command_injection(cls, text: str) -> List[str]:
        """Detect potential command injection attempts."""
        detected = []
        
        for pattern in cls.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"Command injection pattern detected: {pattern}")
        
        return detected
    
    @classmethod
    def sanitize_html(cls, text: str) -> str:
        """Sanitize HTML content."""
        # Escape HTML entities
        sanitized = html.escape(text)
        
        # Remove potentially dangerous tags
        dangerous_tags = [
            r"<script[^>]*>.*?</script>",
            r"<iframe[^>]*>.*?</iframe>",
            r"<object[^>]*>.*?</object>",
            r"<embed[^>]*>.*?</embed>",
            r"<link[^>]*>",
            r"<meta[^>]*>",
        ]
        
        for tag_pattern in dangerous_tags:
            sanitized = re.sub(tag_pattern, "", sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized
    
    @classmethod
    def validate_medical_text(cls, text: str) -> ValidationResult:
        """Validate medical text for security and content appropriateness."""
        errors = []
        warnings = []
        
        # Check for injection attempts
        sql_issues = cls.detect_sql_injection(text)
        xss_issues = cls.detect_xss(text)
        cmd_issues = cls.detect_command_injection(text)
        
        errors.extend([{"type": "sql_injection", "message": issue} for issue in sql_issues])
        errors.extend([{"type": "xss", "message": issue} for issue in xss_issues])
        errors.extend([{"type": "command_injection", "message": issue} for issue in cmd_issues])
        
        # Check text length
        if len(text) > 50000:  # 50KB limit
            errors.append({
                "type": "length_limit",
                "message": f"Text too long: {len(text)} characters (max: 50000)"
            })
        
        # Check for suspicious patterns
        suspicious_patterns = [
            (r"\b(password|passwd|pwd)\s*[:=]\s*\S+", "Potential password exposure"),
            (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "Potential credit card number"),
            (r"\b\d{3}-\d{2}-\d{4}\b", "Potential SSN"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "Email address detected"),
        ]
        
        for pattern, message in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                warnings.append({
                    "type": "sensitive_data",
                    "message": message
                })
        
        # Sanitize the text
        sanitized_text = cls.sanitize_html(text)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_text,
            metadata={
                "original_length": len(text),
                "sanitized_length": len(sanitized_text),
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        )


class DataValidator:
    """Data type and structure validation."""
    
    @staticmethod
    def validate_prescription_text(text: str) -> ValidationResult:
        """Validate prescription text input."""
        errors = []
        warnings = []
        
        # Basic validation
        if not text or not text.strip():
            errors.append({
                "type": "empty_input",
                "message": "Prescription text cannot be empty"
            })
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Length validation
        text = text.strip()
        if len(text) < 10:
            warnings.append({
                "type": "short_text",
                "message": "Prescription text is very short, may not contain sufficient information"
            })
        
        if len(text) > 10000:
            errors.append({
                "type": "text_too_long",
                "message": "Prescription text exceeds maximum length (10,000 characters)"
            })
        
        # Medical content validation
        medical_keywords = [
            "mg", "ml", "tablet", "capsule", "dose", "dosage", "daily", "twice", "thrice",
            "morning", "evening", "night", "before", "after", "meal", "food", "empty stomach",
            "prescription", "medicine", "medication", "drug", "treatment", "therapy"
        ]
        
        text_lower = text.lower()
        found_keywords = [kw for kw in medical_keywords if kw in text_lower]
        
        if len(found_keywords) < 2:
            warnings.append({
                "type": "medical_content",
                "message": "Text may not contain typical medical/prescription content"
            })
        
        # Security validation
        security_result = SecurityValidator.validate_medical_text(text)
        errors.extend(security_result.errors)
        warnings.extend(security_result.warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=security_result.sanitized_data,
            metadata={
                "medical_keywords_found": found_keywords,
                "keyword_count": len(found_keywords),
                **security_result.metadata
            }
        )
    
    @staticmethod
    def validate_user_credentials(credentials: Dict[str, Any]) -> ValidationResult:
        """Validate user credential data."""
        errors = []
        warnings = []
        
        required_fields = ["license_number", "specialization"]
        for field in required_fields:
            if field not in credentials or not credentials[field]:
                errors.append({
                    "type": "missing_field",
                    "message": f"Required field '{field}' is missing or empty"
                })
        
        # Validate license number format
        if "license_number" in credentials:
            license_num = credentials["license_number"]
            if not re.match(r"^[A-Z0-9]{6,20}$", license_num):
                errors.append({
                    "type": "invalid_format",
                    "message": "License number must be 6-20 alphanumeric characters"
                })
        
        # Validate specialization
        if "specialization" in credentials:
            specialization = credentials["specialization"]
            valid_specializations = [
                "General Medicine", "Internal Medicine", "Ayurveda", "Integrative Medicine",
                "Family Medicine", "Cardiology", "Neurology", "Oncology", "Pediatrics",
                "Psychiatry", "Dermatology", "Orthopedics", "Gynecology", "Ophthalmology"
            ]
            
            if specialization not in valid_specializations:
                warnings.append({
                    "type": "specialization_validation",
                    "message": f"Specialization '{specialization}' not in standard list"
                })
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=credentials,
            metadata={
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @staticmethod
    def validate_batch_request(batch_data: List[Dict[str, Any]], max_batch_size: int = 50) -> ValidationResult:
        """Validate batch request data."""
        errors = []
        warnings = []
        
        # Check batch size
        if len(batch_data) > max_batch_size:
            errors.append({
                "type": "batch_size_exceeded",
                "message": f"Batch size {len(batch_data)} exceeds maximum {max_batch_size}"
            })
        
        if len(batch_data) == 0:
            errors.append({
                "type": "empty_batch",
                "message": "Batch request cannot be empty"
            })
        
        # Validate individual items
        item_errors = []
        for i, item in enumerate(batch_data):
            if not isinstance(item, dict):
                item_errors.append({
                    "index": i,
                    "type": "invalid_type",
                    "message": f"Item {i} must be a dictionary"
                })
                continue
            
            if "text" not in item:
                item_errors.append({
                    "index": i,
                    "type": "missing_text",
                    "message": f"Item {i} missing required 'text' field"
                })
        
        if item_errors:
            errors.extend(item_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=batch_data,
            metadata={
                "batch_size": len(batch_data),
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        )


class ResponseValidator:
    """Response validation and sanitization."""
    
    @staticmethod
    def validate_api_response(response_data: Dict[str, Any], user_role: str) -> ValidationResult:
        """Validate API response data based on user role."""
        errors = []
        warnings = []
        sanitized_data = response_data.copy()
        
        # Role-based field filtering
        if user_role == "general_user":
            restricted_fields = [
                "dosage_details", "detailed_dosage", "clinical_contraindications",
                "detailed_interactions", "herb_drug_interactions", "clinical_research_data",
                "research_citations", "contraindication_details", "safety_profile_detailed",
                "pharmacokinetic_data", "pharmacodynamic_data", "clinical_trials",
                "adverse_reactions_detailed", "treatment_protocols", "patient_specific_data"
            ]
            
            removed_fields = []
            for field in restricted_fields:
                if field in sanitized_data:
                    del sanitized_data[field]
                    removed_fields.append(field)
            
            if removed_fields:
                warnings.append({
                    "type": "field_restriction",
                    "message": f"Restricted fields removed for general user: {removed_fields}"
                })
            
            # Add required disclaimers
            sanitized_data["disclaimer"] = (
                "This information is for educational purposes only and is not "
                "personalized medical advice. Consult with qualified healthcare "
                "professionals before making any medical decisions."
            )
            
            sanitized_data["safety_notice"] = (
                "Always consult with a qualified healthcare practitioner before "
                "starting any new treatment or making changes to existing treatments."
            )
        
        # Validate response structure
        if "status" not in response_data:
            warnings.append({
                "type": "missing_status",
                "message": "Response missing status field"
            })
            sanitized_data["status"] = "success"
        
        # Sanitize text fields
        text_fields = ["message", "description", "explanation", "reasoning"]
        for field in text_fields:
            if field in sanitized_data and isinstance(sanitized_data[field], str):
                sanitized_data[field] = SecurityValidator.sanitize_html(sanitized_data[field])
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized_data,
            metadata={
                "user_role": user_role,
                "validation_timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @staticmethod
    def add_security_metadata(response_data: Dict[str, Any], request: Request) -> Dict[str, Any]:
        """Add security metadata to response."""
        enhanced_response = response_data.copy()
        
        # Add security headers information
        enhanced_response["_security"] = {
            "request_id": getattr(request.state, "request_id", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "validation_applied": True,
            "sanitization_applied": True
        }
        
        return enhanced_response


class RequestSanitizer:
    """Request data sanitization utilities."""
    
    @staticmethod
    def sanitize_request_data(data: Any) -> Any:
        """Recursively sanitize request data."""
        if isinstance(data, str):
            # Remove null bytes and control characters
            sanitized = data.replace('\x00', '').replace('\r', '').replace('\n', ' ')
            
            # Limit string length
            if len(sanitized) > 10000:
                sanitized = sanitized[:10000]
            
            # HTML escape
            sanitized = html.escape(sanitized)
            
            return sanitized.strip()
        
        elif isinstance(data, dict):
            return {
                key: RequestSanitizer.sanitize_request_data(value)
                for key, value in data.items()
                if key not in ["__proto__", "constructor", "prototype"]  # Prototype pollution protection
            }
        
        elif isinstance(data, list):
            # Limit list size
            if len(data) > 1000:
                data = data[:1000]
            
            return [RequestSanitizer.sanitize_request_data(item) for item in data]
        
        elif isinstance(data, (int, float)):
            # Validate numeric ranges
            if isinstance(data, float) and (data != data or data == float('inf') or data == float('-inf')):
                return 0.0  # Replace NaN and infinity with 0
            
            return data
        
        else:
            return data


# Pydantic models for validation
class PrescriptionValidationRequest(BaseModel):
    """Validation model for prescription requests."""
    text: str = Field(..., min_length=1, max_length=10000, description="Prescription text")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('text')
    def validate_prescription_text(cls, v):
        result = DataValidator.validate_prescription_text(v)
        if not result.is_valid:
            raise ValueError(f"Invalid prescription text: {result.errors}")
        return result.sanitized_data


class BatchValidationRequest(BaseModel):
    """Validation model for batch requests."""
    items: List[Dict[str, Any]] = Field(..., max_items=50, description="Batch items")
    
    @validator('items')
    def validate_batch_items(cls, v):
        result = DataValidator.validate_batch_request(v)
        if not result.is_valid:
            raise ValueError(f"Invalid batch request: {result.errors}")
        return result.sanitized_data


class CredentialsValidationRequest(BaseModel):
    """Validation model for credentials."""
    license_number: str = Field(..., min_length=6, max_length=20, description="License number")
    specialization: str = Field(..., min_length=1, max_length=100, description="Medical specialization")
    verification_status: bool = Field(default=False, description="Verification status")
    expiry_date: Optional[datetime] = Field(None, description="Credential expiry date")
    
    @validator('license_number')
    def validate_license_format(cls, v):
        if not re.match(r"^[A-Z0-9]{6,20}$", v):
            raise ValueError("License number must be 6-20 alphanumeric characters")
        return v


# Middleware integration functions
def create_validation_middleware():
    """Create validation middleware for FastAPI."""
    
    async def validation_middleware(request: Request, call_next):
        """Validation middleware function."""
        try:
            # Skip validation for certain paths
            skip_paths = ["/docs", "/redoc", "/openapi.json", "/health", "/"]
            if request.url.path in skip_paths:
                return await call_next(request)
            
            # Validate request size
            content_length = request.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:  # 10MB
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Request too large"
                )
            
            # Validate content type for POST/PUT requests
            if request.method in ["POST", "PUT"]:
                content_type = request.headers.get('content-type', '').split(';')[0]
                allowed_types = ["application/json", "application/x-www-form-urlencoded", "multipart/form-data"]
                if content_type not in allowed_types:
                    raise HTTPException(
                        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail=f"Unsupported content type: {content_type}"
                    )
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            
            return response
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Validation middleware error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Request validation failed"
            )
    
    return validation_middleware


# Utility functions
def validate_and_sanitize_input(data: Any, validation_type: str = "general") -> ValidationResult:
    """Validate and sanitize input data based on type."""
    if validation_type == "prescription":
        if isinstance(data, str):
            return DataValidator.validate_prescription_text(data)
        else:
            return ValidationResult(
                is_valid=False,
                errors=[{"type": "invalid_type", "message": "Prescription data must be string"}],
                warnings=[]
            )
    
    elif validation_type == "credentials":
        if isinstance(data, dict):
            return DataValidator.validate_user_credentials(data)
        else:
            return ValidationResult(
                is_valid=False,
                errors=[{"type": "invalid_type", "message": "Credentials data must be dictionary"}],
                warnings=[]
            )
    
    elif validation_type == "batch":
        if isinstance(data, list):
            return DataValidator.validate_batch_request(data)
        else:
            return ValidationResult(
                is_valid=False,
                errors=[{"type": "invalid_type", "message": "Batch data must be list"}],
                warnings=[]
            )
    
    else:
        # General sanitization
        sanitized = RequestSanitizer.sanitize_request_data(data)
        return ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            sanitized_data=sanitized,
            metadata={"validation_type": validation_type}
        )