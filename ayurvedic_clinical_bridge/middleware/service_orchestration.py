"""
Service orchestration middleware for managing inter-service communication,
error handling, request/response validation, and security measures.

This module provides comprehensive orchestration capabilities including:
- Service-to-service communication management
- Centralized error handling and user feedback
- Request/response validation and sanitization
- Rate limiting and security measures
- Circuit breaker patterns for resilience
- Request tracing and monitoring

Requirements: All requirements - system reliability
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, ValidationError, Field
import httpx
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


class ServiceStatus(Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ServiceHealth:
    """Service health information."""
    name: str
    status: ServiceStatus
    last_check: datetime
    response_time: Optional[float] = None
    error_rate: float = 0.0
    uptime: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitBreakerState:
    """Circuit breaker state management."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "closed"  # closed, open, half_open
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half_open -> closed transition


class ServiceRegistry:
    """Registry for managing service endpoints and health."""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.health_status: Dict[str, ServiceHealth] = {}
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
    def register_service(
        self,
        name: str,
        endpoint: str,
        health_check_path: str = "/health",
        timeout: float = 30.0,
        retry_count: int = 3
    ):
        """Register a service with the registry."""
        self.services[name] = {
            "endpoint": endpoint,
            "health_check_path": health_check_path,
            "timeout": timeout,
            "retry_count": retry_count,
            "registered_at": datetime.utcnow()
        }
        self.circuit_breakers[name] = CircuitBreakerState()
        logger.info(f"Service registered: {name} at {endpoint}")
    
    def get_service_endpoint(self, name: str) -> Optional[str]:
        """Get service endpoint by name."""
        service = self.services.get(name)
        return service["endpoint"] if service else None
    
    def is_service_healthy(self, name: str) -> bool:
        """Check if service is healthy."""
        health = self.health_status.get(name)
        if not health:
            return False
        return health.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    
    def update_health_status(self, name: str, health: ServiceHealth):
        """Update service health status."""
        self.health_status[name] = health
    
    def get_circuit_breaker(self, name: str) -> CircuitBreakerState:
        """Get circuit breaker state for service."""
        return self.circuit_breakers.get(name, CircuitBreakerState())


class RequestValidator:
    """Request validation and sanitization."""
    
    @staticmethod
    def sanitize_input(data: Any) -> Any:
        """Sanitize input data to prevent injection attacks."""
        if isinstance(data, str):
            # Basic sanitization - remove potentially dangerous characters
            dangerous_chars = ['<', '>', '"', "'", '&', '\x00', '\n', '\r', '\t']
            sanitized = data
            for char in dangerous_chars:
                sanitized = sanitized.replace(char, '')
            return sanitized.strip()
        elif isinstance(data, dict):
            return {key: RequestValidator.sanitize_input(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [RequestValidator.sanitize_input(item) for item in data]
        return data
    
    @staticmethod
    def validate_request_size(request: Request, max_size: int = 10 * 1024 * 1024):  # 10MB default
        """Validate request size."""
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request too large. Maximum size: {max_size} bytes"
            )
    
    @staticmethod
    def validate_content_type(request: Request, allowed_types: List[str]):
        """Validate request content type."""
        content_type = request.headers.get('content-type', '').split(';')[0]
        if content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported content type. Allowed: {allowed_types}"
            )


class ErrorHandler:
    """Centralized error handling and user feedback."""
    
    @staticmethod
    def create_error_response(
        error: Exception,
        request: Request,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        user_message: Optional[str] = None,
        include_details: bool = False
    ) -> JSONResponse:
        """Create standardized error response."""
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Determine status code and message
        if isinstance(error, HTTPException):
            status_code = error.status_code
            error_message = error.detail
        elif isinstance(error, ValidationError):
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
            error_message = "Validation error"
        elif isinstance(error, TimeoutError):
            status_code = status.HTTP_504_GATEWAY_TIMEOUT
            error_message = "Service timeout"
        elif isinstance(error, ConnectionError):
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            error_message = "Service unavailable"
        else:
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
            error_message = "Internal server error"
        
        # Create error response
        error_response = {
            "error": {
                "type": type(error).__name__,
                "message": user_message or error_message,
                "severity": severity.value,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat(),
                "path": str(request.url.path),
                "method": request.method
            }
        }
        
        # Add details for debugging (only in development or for high-severity errors)
        if include_details or severity == ErrorSeverity.CRITICAL:
            error_response["error"]["details"] = str(error)
            error_response["error"]["error_class"] = type(error).__name__
        
        # Log error
        log_level = {
            ErrorSeverity.LOW: logger.info,
            ErrorSeverity.MEDIUM: logger.warning,
            ErrorSeverity.HIGH: logger.error,
            ErrorSeverity.CRITICAL: logger.critical
        }[severity]
        
        log_level(f"Error {request_id}: {error_message} - {str(error)}")
        
        return JSONResponse(
            status_code=status_code,
            content=error_response
        )
    
    @staticmethod
    def handle_service_error(
        service_name: str,
        error: Exception,
        request: Request,
        fallback_response: Optional[Dict[str, Any]] = None
    ) -> JSONResponse:
        """Handle service-specific errors with fallback options."""
        request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
        
        # Log service error
        logger.error(f"Service error in {service_name}: {str(error)} (Request: {request_id})")
        
        # If fallback response is available, use it
        if fallback_response:
            logger.info(f"Using fallback response for {service_name} (Request: {request_id})")
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "data": fallback_response,
                    "metadata": {
                        "fallback_used": True,
                        "original_service": service_name,
                        "request_id": request_id,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    "warning": f"Primary service {service_name} unavailable, using cached/fallback data"
                }
            )
        
        # Otherwise, return error response
        return ErrorHandler.create_error_response(
            error=error,
            request=request,
            severity=ErrorSeverity.HIGH,
            user_message=f"Service {service_name} is currently unavailable. Please try again later."
        )


class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self):
        self.limiter = Limiter(key_func=get_remote_address)
        self.custom_limits: Dict[str, str] = {}
    
    def set_rate_limit(self, endpoint: str, limit: str):
        """Set custom rate limit for endpoint."""
        self.custom_limits[endpoint] = limit
    
    def get_rate_limit(self, endpoint: str) -> str:
        """Get rate limit for endpoint."""
        return self.custom_limits.get(endpoint, "100/minute")


class ServiceOrchestrator:
    """Main service orchestration class."""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.rate_limiter = RateLimiter()
        self.request_timeout = 30.0
        self.max_retries = 3
        self.health_check_interval = 60  # seconds
        self._setup_default_services()
        self._setup_rate_limits()
    
    def _setup_default_services(self):
        """Setup default service registrations."""
        # Register core services
        services = [
            ("prescription_parser", "http://localhost:8001", "/health"),
            ("semantic_mapper", "http://localhost:8002", "/health"),
            ("safety_analyzer", "http://localhost:8003", "/health"),
            ("recommendation_engine", "http://localhost:8004", "/health"),
            ("knowledge_base", "http://localhost:8005", "/health"),
            ("neo4j", "http://localhost:7474", "/db/data/"),
            ("chromadb", "http://localhost:8000", "/api/v1/heartbeat")
        ]
        
        for name, endpoint, health_path in services:
            self.registry.register_service(name, endpoint, health_path)
    
    def _setup_rate_limits(self):
        """Setup default rate limits."""
        rate_limits = {
            "/auth/login": "5/minute",
            "/auth/register": "3/minute",
            "/prescription/parse": "50/minute",
            "/prescription/batch-parse": "5/minute",
            "/semantic-mapping/map": "30/minute",
            "/semantic-mapping/batch-map": "3/minute",
            "/safety-analysis/analyze": "100/minute",
            "/safety-analysis/batch-analyze": "10/minute",
            "/recommendations/generate": "20/minute",
            "/recommendations/batch-generate": "2/minute"
        }
        
        for endpoint, limit in rate_limits.items():
            self.rate_limiter.set_rate_limit(endpoint, limit)
    
    async def call_service(
        self,
        service_name: str,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        fallback_response: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make a call to a registered service with circuit breaker pattern."""
        circuit_breaker = self.registry.get_circuit_breaker(service_name)
        
        # Check circuit breaker state
        if circuit_breaker.state == "open":
            if (datetime.utcnow() - circuit_breaker.last_failure_time).seconds < circuit_breaker.recovery_timeout:
                if fallback_response:
                    logger.warning(f"Circuit breaker open for {service_name}, using fallback")
                    return fallback_response
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Service {service_name} is currently unavailable (circuit breaker open)"
                )
            else:
                # Move to half-open state
                circuit_breaker.state = "half_open"
                logger.info(f"Circuit breaker for {service_name} moved to half-open state")
        
        endpoint = self.registry.get_service_endpoint(service_name)
        if not endpoint:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service {service_name} not registered"
            )
        
        url = f"{endpoint.rstrip('/')}/{path.lstrip('/')}"
        timeout = timeout or self.request_timeout
        
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                start_time = time.time()
                
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, json=data, headers=headers)
                elif method.upper() == "PUT":
                    response = await client.put(url, json=data, headers=headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response_time = time.time() - start_time
                
                # Handle response
                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Service {service_name} returned error: {response.text}"
                    )
                
                # Success - update circuit breaker
                if circuit_breaker.state == "half_open":
                    circuit_breaker.failure_count = 0
                    circuit_breaker.state = "closed"
                    logger.info(f"Circuit breaker for {service_name} closed after successful call")
                
                # Update health status
                health = ServiceHealth(
                    name=service_name,
                    status=ServiceStatus.HEALTHY,
                    last_check=datetime.utcnow(),
                    response_time=response_time
                )
                self.registry.update_health_status(service_name, health)
                
                return response.json()
                
        except Exception as e:
            # Handle failure - update circuit breaker
            circuit_breaker.failure_count += 1
            circuit_breaker.last_failure_time = datetime.utcnow()
            
            if circuit_breaker.failure_count >= circuit_breaker.failure_threshold:
                circuit_breaker.state = "open"
                logger.error(f"Circuit breaker opened for {service_name} after {circuit_breaker.failure_count} failures")
            
            # Update health status
            health = ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
            self.registry.update_health_status(service_name, health)
            
            # Use fallback if available
            if fallback_response:
                logger.warning(f"Service {service_name} failed, using fallback response")
                return fallback_response
            
            raise e
    
    async def health_check_service(self, service_name: str) -> ServiceHealth:
        """Perform health check on a service."""
        try:
            service_info = self.registry.services.get(service_name)
            if not service_info:
                return ServiceHealth(
                    name=service_name,
                    status=ServiceStatus.UNKNOWN,
                    last_check=datetime.utcnow(),
                    metadata={"error": "Service not registered"}
                )
            
            endpoint = service_info["endpoint"]
            health_path = service_info["health_check_path"]
            url = f"{endpoint.rstrip('/')}/{health_path.lstrip('/')}"
            
            start_time = time.time()
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    status = ServiceStatus.HEALTHY
                elif response.status_code < 500:
                    status = ServiceStatus.DEGRADED
                else:
                    status = ServiceStatus.UNHEALTHY
                
                health = ServiceHealth(
                    name=service_name,
                    status=status,
                    last_check=datetime.utcnow(),
                    response_time=response_time,
                    metadata=response.json() if response.headers.get('content-type', '').startswith('application/json') else {}
                )
                
                self.registry.update_health_status(service_name, health)
                return health
                
        except Exception as e:
            health = ServiceHealth(
                name=service_name,
                status=ServiceStatus.UNHEALTHY,
                last_check=datetime.utcnow(),
                metadata={"error": str(e)}
            )
            self.registry.update_health_status(service_name, health)
            return health
    
    async def health_check_all_services(self) -> Dict[str, ServiceHealth]:
        """Perform health check on all registered services."""
        tasks = []
        for service_name in self.registry.services.keys():
            tasks.append(self.health_check_service(service_name))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = {}
        for i, result in enumerate(results):
            service_name = list(self.registry.services.keys())[i]
            if isinstance(result, Exception):
                health_status[service_name] = ServiceHealth(
                    name=service_name,
                    status=ServiceStatus.UNHEALTHY,
                    last_check=datetime.utcnow(),
                    metadata={"error": str(result)}
                )
            else:
                health_status[service_name] = result
        
        return health_status
    
    def get_service_status_summary(self) -> Dict[str, Any]:
        """Get summary of all service statuses."""
        summary = {
            "total_services": len(self.registry.services),
            "healthy_services": 0,
            "degraded_services": 0,
            "unhealthy_services": 0,
            "unknown_services": 0,
            "services": {}
        }
        
        for service_name, health in self.registry.health_status.items():
            summary["services"][service_name] = {
                "status": health.status.value,
                "last_check": health.last_check.isoformat(),
                "response_time": health.response_time,
                "error_rate": health.error_rate
            }
            
            if health.status == ServiceStatus.HEALTHY:
                summary["healthy_services"] += 1
            elif health.status == ServiceStatus.DEGRADED:
                summary["degraded_services"] += 1
            elif health.status == ServiceStatus.UNHEALTHY:
                summary["unhealthy_services"] += 1
            else:
                summary["unknown_services"] += 1
        
        return summary


# Global orchestrator instance
orchestrator = ServiceOrchestrator()


# Middleware functions
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


async def request_validation_middleware(request: Request, call_next):
    """Validate and sanitize requests."""
    try:
        # Validate request size
        RequestValidator.validate_request_size(request)
        
        # Validate content type for POST/PUT requests
        if request.method in ["POST", "PUT"]:
            allowed_types = ["application/json", "application/x-www-form-urlencoded", "multipart/form-data"]
            RequestValidator.validate_content_type(request, allowed_types)
        
        response = await call_next(request)
        return response
        
    except HTTPException as e:
        return ErrorHandler.create_error_response(e, request, ErrorSeverity.MEDIUM)
    except Exception as e:
        return ErrorHandler.create_error_response(e, request, ErrorSeverity.HIGH)


async def error_handling_middleware(request: Request, call_next):
    """Global error handling middleware."""
    try:
        response = await call_next(request)
        return response
    except HTTPException as e:
        return ErrorHandler.create_error_response(e, request, ErrorSeverity.MEDIUM)
    except ValidationError as e:
        return ErrorHandler.create_error_response(e, request, ErrorSeverity.MEDIUM, "Invalid request data")
    except TimeoutError as e:
        return ErrorHandler.create_error_response(e, request, ErrorSeverity.HIGH, "Request timeout")
    except ConnectionError as e:
        return ErrorHandler.create_error_response(e, request, ErrorSeverity.HIGH, "Service connection error")
    except Exception as e:
        return ErrorHandler.create_error_response(e, request, ErrorSeverity.CRITICAL, "Internal server error")


async def security_headers_middleware(request: Request, call_next):
    """Add security headers to responses."""
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response


# Utility functions for service communication
async def call_prescription_service(
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    fallback_response: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Call prescription parsing service."""
    return await orchestrator.call_service(
        "prescription_parser", method, path, data, fallback_response=fallback_response
    )


async def call_semantic_mapping_service(
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    fallback_response: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Call semantic mapping service."""
    return await orchestrator.call_service(
        "semantic_mapper", method, path, data, fallback_response=fallback_response
    )


async def call_safety_analysis_service(
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    fallback_response: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Call safety analysis service."""
    return await orchestrator.call_service(
        "safety_analyzer", method, path, data, fallback_response=fallback_response
    )


async def call_recommendation_service(
    method: str,
    path: str,
    data: Optional[Dict[str, Any]] = None,
    fallback_response: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Call recommendation engine service."""
    return await orchestrator.call_service(
        "recommendation_engine", method, path, data, fallback_response=fallback_response
    )


# Health check endpoints
async def get_orchestrator_health() -> Dict[str, Any]:
    """Get orchestrator health status."""
    service_health = await orchestrator.health_check_all_services()
    summary = orchestrator.get_service_status_summary()
    
    overall_status = "healthy"
    if summary["unhealthy_services"] > 0:
        overall_status = "unhealthy"
    elif summary["degraded_services"] > 0:
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator": "operational",
        "services": summary,
        "circuit_breakers": {
            name: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            for name, cb in orchestrator.registry.circuit_breakers.items()
        }
    }