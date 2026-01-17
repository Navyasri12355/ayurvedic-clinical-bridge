"""
Authentication API routes for user management and JWT authentication.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials
from loguru import logger

from ayurvedic_clinical_bridge.models.user_models import (
    User, UserCreate, UserLogin, UserResponse, Token, PractitionerCredentials
)
from ayurvedic_clinical_bridge.services.auth_service import auth_service
from ayurvedic_clinical_bridge.middleware.auth_middleware import (
    get_current_active_user, require_qualified_practitioner, AccessControl, AuditLogger
)


# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_create: UserCreate):
    """Register a new user."""
    try:
        user = auth_service.create_user(user_create)
        logger.info(f"User registered: {user.email}")
        
        # Return user response (excluding sensitive data)
        return UserResponse(
            id=user.id,
            email=user.email,
            role=user.role,
            credentials=user.credentials,
            preferences=user.preferences,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login_user(user_login: UserLogin):
    """Login user and return JWT token."""
    try:
        token = auth_service.login_user(user_login)
        logger.info(f"User logged in: {user_login.email}")
        return token
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role,
        credentials=current_user.credentials,
        preferences=current_user.preferences,
        is_active=current_user.is_active,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )


@router.post("/credentials", response_model=UserResponse)
async def update_practitioner_credentials(
    credentials: PractitionerCredentials,
    current_user: User = Depends(get_current_active_user)
):
    """Update practitioner credentials for current user."""
    try:
        updated_user = auth_service.update_user_credentials(current_user.id, credentials)
        logger.info(f"Credentials updated for user: {current_user.email}")
        
        return UserResponse(
            id=updated_user.id,
            email=updated_user.email,
            role=updated_user.role,
            credentials=updated_user.credentials,
            preferences=updated_user.preferences,
            is_active=updated_user.is_active,
            created_at=updated_user.created_at,
            last_login=updated_user.last_login
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Credential update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Credential update failed"
        )


@router.get("/verify-practitioner")
async def verify_practitioner_status(current_user: User = Depends(require_qualified_practitioner)):
    """Verify practitioner status (requires qualified practitioner credentials)."""
    return {
        "verified": True,
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "credentials": {
            "license_number": current_user.credentials.license_number,
            "specialization": current_user.credentials.specialization,
            "verification_status": current_user.credentials.verification_status,
            "expiry_date": current_user.credentials.expiry_date
        }
    }


@router.post("/logout")
async def logout_user(current_user: User = Depends(get_current_active_user)):
    """Logout user (client-side token removal)."""
    # In a stateless JWT system, logout is typically handled client-side
    # by removing the token. For enhanced security, you could implement
    # a token blacklist or use refresh tokens.
    logger.info(f"User logged out: {current_user.email}")
    return {"message": "Successfully logged out"}


@router.get("/access-level")
async def get_user_access_level(
    request: Request,
    current_user: User = Depends(get_current_active_user)
):
    """Get user access level and permissions."""
    access_level = AccessControl.get_user_access_level(current_user)
    permissions = AccessControl.get_user_permissions(current_user)
    
    # Log access level check
    AuditLogger.log_feature_access(
        user=current_user,
        feature="access_level_check",
        granted=True,
        request=request
    )
    
    return {
        "user_id": current_user.id,
        "email": current_user.email,
        "role": current_user.role,
        "access_level": access_level,
        "permissions": permissions
    }