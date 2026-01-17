"""
Admin API routes for managing practitioner verifications and system administration.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from loguru import logger

from ayurvedic_clinical_bridge.models.user_models import User, UserResponse
from ayurvedic_clinical_bridge.services.auth_service import auth_service
from ayurvedic_clinical_bridge.middleware.auth_middleware import get_current_active_user


# Create router
router = APIRouter(prefix="/admin", tags=["administration"])


def require_admin_access(current_user: User = Depends(get_current_active_user)):
    """Require admin access (for demo purposes, any qualified practitioner can access)."""
    # In production, you would have a separate admin role
    # For now, we'll allow any qualified practitioner to access admin functions
    if current_user.role.value != "qualified_practitioner":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


@router.get("/pending-practitioners", response_model=List[UserResponse])
async def get_pending_practitioners(admin_user: User = Depends(require_admin_access)):
    """Get list of practitioners pending verification."""
    try:
        pending_practitioners = []
        
        # Get all users with practitioner role and unverified credentials
        for user in auth_service.users_db.values():
            if (user.role.value == "qualified_practitioner" and 
                user.credentials and 
                not user.credentials.verification_status):
                
                pending_practitioners.append(UserResponse(
                    id=user.id,
                    email=user.email,
                    role=user.role,
                    credentials=user.credentials,
                    preferences=user.preferences,
                    is_active=user.is_active,
                    created_at=user.created_at,
                    last_login=user.last_login
                ))
        
        logger.info(f"Retrieved {len(pending_practitioners)} pending practitioners")
        return pending_practitioners
        
    except Exception as e:
        logger.error(f"Failed to get pending practitioners: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pending practitioners"
        )


@router.post("/verify-practitioner")
async def verify_practitioner(
    verification_request: dict,
    admin_user: User = Depends(require_admin_access)
):
    """Verify or reject a practitioner's credentials."""
    try:
        practitioner_id = verification_request.get("practitioner_id")
        approved = verification_request.get("approved", False)
        
        if not practitioner_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Practitioner ID is required"
            )
        
        # Get the practitioner
        practitioner = auth_service.get_user_by_id(practitioner_id)
        if not practitioner:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Practitioner not found"
            )
        
        if not practitioner.credentials:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Practitioner has no credentials to verify"
            )
        
        # Update verification status
        practitioner.credentials.verification_status = approved
        
        action = "approved" if approved else "rejected"
        logger.info(f"Practitioner {practitioner.email} credentials {action} by {admin_user.email}")
        
        return {
            "message": f"Practitioner credentials {action} successfully",
            "practitioner_id": practitioner_id,
            "approved": approved
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to verify practitioner: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process verification"
        )


@router.get("/system-stats")
async def get_system_stats(admin_user: User = Depends(require_admin_access)):
    """Get system statistics for admin dashboard."""
    try:
        total_users = len(auth_service.users_db)
        general_users = sum(1 for user in auth_service.users_db.values() 
                           if user.role.value == "general_user")
        practitioners = sum(1 for user in auth_service.users_db.values() 
                           if user.role.value == "qualified_practitioner")
        verified_practitioners = sum(1 for user in auth_service.users_db.values() 
                                   if (user.role.value == "qualified_practitioner" and 
                                       user.credentials and 
                                       user.credentials.verification_status))
        pending_verifications = practitioners - verified_practitioners
        
        stats = {
            "total_users": total_users,
            "general_users": general_users,
            "practitioners": practitioners,
            "verified_practitioners": verified_practitioners,
            "pending_verifications": pending_verifications,
            "active_users": sum(1 for user in auth_service.users_db.values() if user.is_active)
        }
        
        logger.info(f"System stats retrieved by {admin_user.email}")
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )


@router.get("/users", response_model=List[UserResponse])
async def get_all_users(admin_user: User = Depends(require_admin_access)):
    """Get list of all users (admin only)."""
    try:
        users = []
        for user in auth_service.users_db.values():
            users.append(UserResponse(
                id=user.id,
                email=user.email,
                role=user.role,
                credentials=user.credentials,
                preferences=user.preferences,
                is_active=user.is_active,
                created_at=user.created_at,
                last_login=user.last_login
            ))
        
        logger.info(f"All users list retrieved by {admin_user.email}")
        return users
        
    except Exception as e:
        logger.error(f"Failed to get users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.post("/toggle-user-status")
async def toggle_user_status(
    request: dict,
    admin_user: User = Depends(require_admin_access)
):
    """Toggle user active status (admin only)."""
    try:
        user_id = request.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User ID is required"
            )
        
        user = auth_service.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Toggle active status
        user.is_active = not user.is_active
        
        status_text = "activated" if user.is_active else "deactivated"
        logger.info(f"User {user.email} {status_text} by {admin_user.email}")
        
        return {
            "message": f"User {status_text} successfully",
            "user_id": user_id,
            "is_active": user.is_active
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to toggle user status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user status"
        )