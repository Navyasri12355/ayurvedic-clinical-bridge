"""
Authentication service for JWT-based authentication and user management.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
from loguru import logger

from ayurvedic_clinical_bridge.models.user_models import (
    User, UserCreate, UserLogin, Token, TokenData, UserRole, PractitionerCredentials
)


class AuthService:
    """Authentication service for user management and JWT operations."""
    
    def __init__(self):
        """Initialize authentication service."""
        # Initialize password context with explicit bcrypt configuration
        try:
            self.pwd_context = CryptContext(
                schemes=["bcrypt"], 
                deprecated="auto",
                bcrypt__rounds=12  # Explicit rounds configuration
            )
        except Exception as e:
            logger.warning(f"Failed to initialize bcrypt with rounds, using default: {e}")
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # In-memory user storage (replace with database in production)
        self.users_db: Dict[str, User] = {}
        self.email_to_id: Dict[str, str] = {}
        
        logger.info("AuthService initialized")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        try:
            import bcrypt
            
            # Handle password truncation the same way as in hashing
            password_bytes = plain_password.encode('utf-8')
            if len(password_bytes) > 72:
                password_bytes = password_bytes[:72]
                try:
                    plain_password = password_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    for i in range(71, 0, -1):
                        try:
                            plain_password = password_bytes[:i].decode('utf-8')
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        plain_password = ''.join(c for c in plain_password if ord(c) < 128)[:72]
                
                password_bytes = plain_password.encode('utf-8')
            
            # Verify using bcrypt directly
            return bcrypt.checkpw(password_bytes, hashed_password.encode('utf-8'))
            
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False
    
    def get_password_hash(self, password: str) -> str:
        """Generate password hash."""
        try:
            import bcrypt
            
            # Ensure password is within bcrypt's 72-byte limit
            password_bytes = password.encode('utf-8')
            if len(password_bytes) > 72:
                # Truncate safely at byte level
                password_bytes = password_bytes[:72]
                # Try to decode back to ensure we don't have broken UTF-8
                try:
                    password = password_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    # If we broke a UTF-8 sequence, truncate more conservatively
                    for i in range(71, 0, -1):
                        try:
                            password = password_bytes[:i].decode('utf-8')
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        # Last resort: use only ASCII part
                        password = ''.join(c for c in password if ord(c) < 128)[:72]
                
                password_bytes = password.encode('utf-8')
            
            # Generate salt and hash
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password_bytes, salt)
            return hashed.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Password hashing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password hashing system error"
            )
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id: str = payload.get("sub")
            email: str = payload.get("email")
            role: str = payload.get("role")
            
            if user_id is None or email is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            token_data = TokenData(
                user_id=user_id,
                email=email,
                role=UserRole(role) if role else None
            )
            return token_data
            
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address."""
        user_id = self.email_to_id.get(email)
        if user_id:
            return self.users_db.get(user_id)
        return None
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self.users_db.get(user_id)
    
    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password."""
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        if not user.is_active:
            return None
        return user
    
    def create_user(self, user_create: UserCreate) -> User:
        """Create a new user."""
        # Check if user already exists
        if self.get_user_by_email(user_create.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Generate user ID
        user_id = f"user_{len(self.users_db) + 1:06d}"
        
        # Hash password
        hashed_password = self.get_password_hash(user_create.password)
        
        # Verify practitioner credentials if provided
        if user_create.credentials and user_create.role == UserRole.QUALIFIED_PRACTITIONER:
            if not self.verify_practitioner_credentials(user_create.credentials):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid practitioner credentials"
                )
        
        # Create user
        user = User(
            id=user_id,
            email=user_create.email,
            hashed_password=hashed_password,
            role=user_create.role,
            credentials=user_create.credentials
        )
        
        # Store user
        self.users_db[user_id] = user
        self.email_to_id[user_create.email] = user_id
        
        logger.info(f"User created: {user.email} with role {user.role}")
        return user
    
    def login_user(self, user_login: UserLogin) -> Token:
        """Login user and return JWT token."""
        user = self.authenticate_user(user_login.email, user_login.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login
        user.last_login = datetime.now(timezone.utc)
        
        # Create access token
        access_token_expires = timedelta(minutes=self.access_token_expire_minutes)
        access_token = self.create_access_token(
            data={
                "sub": user.id,
                "email": user.email,
                "role": user.role.value
            },
            expires_delta=access_token_expires
        )
        
        logger.info(f"User logged in: {user.email}")
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=self.access_token_expire_minutes * 60
        )
    
    def verify_practitioner_credentials(self, credentials: PractitionerCredentials) -> bool:
        """Verify practitioner credentials (placeholder implementation)."""
        # In production, this would integrate with medical licensing authorities
        # For now, we'll do basic validation
        
        if not credentials.license_number or len(credentials.license_number) < 5:
            return False
        
        if not credentials.specialization:
            return False
        
        # Ensure both datetimes are timezone-aware for comparison
        current_time = datetime.now(timezone.utc)
        expiry_date = credentials.expiry_date
        
        # If expiry_date is timezone-naive, make it timezone-aware (assume UTC)
        if expiry_date.tzinfo is None:
            expiry_date = expiry_date.replace(tzinfo=timezone.utc)
        
        if expiry_date <= current_time:
            return False
        
        # Mark as verified for demo purposes
        credentials.verification_status = True
        logger.info(f"Practitioner credentials verified: {credentials.license_number}")
        return True
    
    def update_user_credentials(self, user_id: str, credentials: PractitionerCredentials) -> User:
        """Update user practitioner credentials."""
        user = self.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify credentials
        if not self.verify_practitioner_credentials(credentials):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid practitioner credentials"
            )
        
        # Update user
        user.credentials = credentials
        user.role = UserRole.QUALIFIED_PRACTITIONER
        
        logger.info(f"User credentials updated: {user.email}")
        return user
    
    def is_qualified_practitioner(self, user: User) -> bool:
        """Check if user is a qualified practitioner with verified credentials."""
        if not (user.role == UserRole.QUALIFIED_PRACTITIONER and 
                user.credentials is not None and 
                user.credentials.verification_status):
            return False
        
        # Ensure both datetimes are timezone-aware for comparison
        current_time = datetime.now(timezone.utc)
        expiry_date = user.credentials.expiry_date
        
        # If expiry_date is timezone-naive, make it timezone-aware (assume UTC)
        if expiry_date.tzinfo is None:
            expiry_date = expiry_date.replace(tzinfo=timezone.utc)
        
        return expiry_date > current_time


# Global auth service instance
auth_service = AuthService()