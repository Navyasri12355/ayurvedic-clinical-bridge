"""
Authentication and authorization middleware for role-based access control.
"""

import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger

from ayurvedic_clinical_bridge.models.user_models import User, UserRole
from ayurvedic_clinical_bridge.services.auth_service import auth_service


# HTTP Bearer token scheme
security = HTTPBearer()


class AuditLogger:
    """Audit logging for access control events."""
    
    @staticmethod
    def log_access_attempt(
        user_id: str,
        email: str,
        role: str,
        resource: str,
        action: str,
        granted: bool,
        request_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Log access control events for audit purposes."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "email": email,
            "role": role,
            "resource": resource,
            "action": action,
            "access_granted": granted,
            "request_ip": request_ip,
            "user_agent": user_agent,
            "additional_info": additional_info or {}
        }
        
        # Log to structured logger
        if granted:
            logger.info(f"ACCESS_GRANTED: {json.dumps(audit_entry)}")
        else:
            logger.warning(f"ACCESS_DENIED: {json.dumps(audit_entry)}")
    
    @staticmethod
    def log_feature_access(
        user: User,
        feature: str,
        granted: bool,
        request: Optional[Request] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Log feature access attempts."""
        request_ip = None
        user_agent = None
        
        if request:
            request_ip = request.client.host if request.client else None
            user_agent = request.headers.get("user-agent")
        
        AuditLogger.log_access_attempt(
            user_id=user.id,
            email=user.email,
            role=user.role.value,
            resource=feature,
            action="access",
            granted=granted,
            request_ip=request_ip,
            user_agent=user_agent,
            additional_info=details
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user from JWT token."""
    token = credentials.credentials
    
    # Verify token
    token_data = auth_service.verify_token(token)
    
    # Get user
    user = auth_service.get_user_by_id(token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def require_role(allowed_roles: List[UserRole]):
    """Decorator factory for role-based access control."""
    def role_checker(
        request: Request,
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        granted = current_user.role in allowed_roles
        
        # Log access attempt
        AuditLogger.log_feature_access(
            user=current_user,
            feature=f"role_restricted_endpoint",
            granted=granted,
            request=request,
            details={"required_roles": [role.value for role in allowed_roles]}
        )
        
        if not granted:
            logger.warning(f"Access denied for user {current_user.email} with role {current_user.role}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    return role_checker


def require_qualified_practitioner(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require qualified practitioner with verified credentials."""
    granted = auth_service.is_qualified_practitioner(current_user)
    
    # Log access attempt
    AuditLogger.log_feature_access(
        user=current_user,
        feature="qualified_practitioner_endpoint",
        granted=granted,
        request=request,
        details={
            "credentials_verified": current_user.credentials.verification_status if current_user.credentials else False,
            "credentials_expired": (
                current_user.credentials.expiry_date <= datetime.utcnow() 
                if current_user.credentials else True
            )
        }
    )
    
    if not granted:
        logger.warning(f"Access denied for user {current_user.email} - not a qualified practitioner")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Qualified practitioner credentials required"
        )
    return current_user


def require_general_user_or_practitioner(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Allow both general users and practitioners."""
    allowed_roles = [UserRole.GENERAL_USER, UserRole.QUALIFIED_PRACTITIONER]
    granted = current_user.role in allowed_roles
    
    # Log access attempt
    AuditLogger.log_feature_access(
        user=current_user,
        feature="general_user_endpoint",
        granted=granted,
        request=request
    )
    
    if not granted:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User access required"
        )
    return current_user


class AccessControl:
    """Access control utility class."""
    
    @staticmethod
    def can_access_detailed_dosage(user: User) -> bool:
        """Check if user can access detailed dosage information."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_access_practitioner_features(user: User) -> bool:
        """Check if user can access practitioner-specific features."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_view_safety_details(user: User) -> bool:
        """Check if user can view detailed safety information."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_access_clinical_data(user: User) -> bool:
        """Check if user can access clinical data and research."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_access_herb_drug_interactions(user: User) -> bool:
        """Check if user can access detailed herb-drug interaction data."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_access_contraindications(user: User) -> bool:
        """Check if user can access detailed contraindication information."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_access_research_citations(user: User) -> bool:
        """Check if user can access detailed research citations."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_modify_treatment_plans(user: User) -> bool:
        """Check if user can modify or create treatment plans."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def can_access_patient_data(user: User) -> bool:
        """Check if user can access patient-specific data."""
        return auth_service.is_qualified_practitioner(user)
    
    @staticmethod
    def get_user_access_level(user: User) -> str:
        """Get user access level string."""
        if auth_service.is_qualified_practitioner(user):
            return "practitioner"
        elif user.role == UserRole.GENERAL_USER:
            return "general"
        else:
            return "restricted"
    
    @staticmethod
    def get_user_permissions(user: User) -> Dict[str, bool]:
        """Get comprehensive user permissions dictionary."""
        return {
            "can_access_detailed_dosage": AccessControl.can_access_detailed_dosage(user),
            "can_access_practitioner_features": AccessControl.can_access_practitioner_features(user),
            "can_view_safety_details": AccessControl.can_view_safety_details(user),
            "can_access_clinical_data": AccessControl.can_access_clinical_data(user),
            "can_access_herb_drug_interactions": AccessControl.can_access_herb_drug_interactions(user),
            "can_access_contraindications": AccessControl.can_access_contraindications(user),
            "can_access_research_citations": AccessControl.can_access_research_citations(user),
            "can_modify_treatment_plans": AccessControl.can_modify_treatment_plans(user),
            "can_access_patient_data": AccessControl.can_access_patient_data(user)
        }
    
    @staticmethod
    def filter_response_for_user(
        user: User, 
        response_data: dict,
        request: Optional[Request] = None
    ) -> dict:
        """Filter response data based on user access level."""
        access_level = AccessControl.get_user_access_level(user)
        
        # Log data access
        AuditLogger.log_feature_access(
            user=user,
            feature="data_filtering",
            granted=True,
            request=request,
            details={
                "access_level": access_level,
                "data_keys": list(response_data.keys())
            }
        )
        
        if access_level == "general":
            # Remove detailed dosage and clinical information for general users
            filtered_data = response_data.copy()
            
            # Remove practitioner-only fields
            restricted_fields = [
                "dosage_details",
                "detailed_dosage",
                "clinical_contraindications",
                "detailed_interactions",
                "herb_drug_interactions",
                "clinical_research_data",
                "research_citations",
                "contraindication_details",
                "safety_profile_detailed",
                "pharmacokinetic_data",
                "pharmacodynamic_data",
                "clinical_trials",
                "adverse_reactions_detailed",
                "treatment_protocols",
                "patient_specific_data"
            ]
            
            removed_fields = []
            for field in restricted_fields:
                if field in filtered_data:
                    del filtered_data[field]
                    removed_fields.append(field)
            
            # Log removed fields for audit
            if removed_fields:
                AuditLogger.log_feature_access(
                    user=user,
                    feature="data_restriction",
                    granted=False,
                    request=request,
                    details={"removed_fields": removed_fields}
                )
            
            # Add educational disclaimers
            filtered_data["disclaimer"] = (
                "This information is for educational purposes only and is not "
                "personalized medical advice. Consult with qualified healthcare "
                "professionals before making any medical decisions."
            )
            
            filtered_data["safety_notice"] = (
                "Always consult with a qualified healthcare practitioner before "
                "starting any new treatment or making changes to existing treatments."
            )
            
            return filtered_data
        
        # Practitioners get full access
        return response_data
    
    @staticmethod
    def check_feature_access(
        user: User,
        feature: str,
        request: Optional[Request] = None,
        raise_on_deny: bool = True
    ) -> bool:
        """Check access to a specific feature and optionally raise exception."""
        feature_permissions = {
            "detailed_dosage": AccessControl.can_access_detailed_dosage,
            "practitioner_features": AccessControl.can_access_practitioner_features,
            "safety_details": AccessControl.can_view_safety_details,
            "clinical_data": AccessControl.can_access_clinical_data,
            "herb_drug_interactions": AccessControl.can_access_herb_drug_interactions,
            "contraindications": AccessControl.can_access_contraindications,
            "research_citations": AccessControl.can_access_research_citations,
            "treatment_plans": AccessControl.can_modify_treatment_plans,
            "patient_data": AccessControl.can_access_patient_data
        }
        
        permission_func = feature_permissions.get(feature)
        if not permission_func:
            # Unknown feature - deny access
            granted = False
        else:
            granted = permission_func(user)
        
        # Log access attempt
        AuditLogger.log_feature_access(
            user=user,
            feature=feature,
            granted=granted,
            request=request
        )
        
        if not granted and raise_on_deny:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied to feature: {feature}"
            )
        
        return granted


# Convenience dependency aliases
RequireGeneralUser = require_role([UserRole.GENERAL_USER])
RequirePractitioner = require_role([UserRole.QUALIFIED_PRACTITIONER])
RequireAnyUser = require_general_user_or_practitioner


# Feature-specific access control decorators
def require_detailed_dosage_access(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require access to detailed dosage information."""
    AccessControl.check_feature_access(current_user, "detailed_dosage", request)
    return current_user


def require_clinical_data_access(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require access to clinical data."""
    AccessControl.check_feature_access(current_user, "clinical_data", request)
    return current_user


def require_safety_details_access(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require access to detailed safety information."""
    AccessControl.check_feature_access(current_user, "safety_details", request)
    return current_user


def require_herb_drug_interaction_access(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require access to herb-drug interaction data."""
    AccessControl.check_feature_access(current_user, "herb_drug_interactions", request)
    return current_user


def require_contraindication_access(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require access to contraindication information."""
    AccessControl.check_feature_access(current_user, "contraindications", request)
    return current_user