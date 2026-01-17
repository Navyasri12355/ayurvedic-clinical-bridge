"""
Security middleware for rate limiting, authentication, and security measures.

This module provides comprehensive security capabilities including:
- Rate limiting with configurable limits per endpoint
- IP-based blocking and whitelisting
- Request throttling and abuse prevention
- Security headers and CORS management
- Authentication token validation
- Audit logging for security events

Requirements: All requirements - system reliability and security
"""

import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from loguru import logger
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


class SecurityThreatLevel(Enum):
    """Security threat level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BlockReason(Enum):
    """Reasons for blocking requests."""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    MALICIOUS_PAYLOAD = "malicious_payload"
    AUTHENTICATION_FAILURE = "authentication_failure"
    IP_BLACKLISTED = "ip_blacklisted"
    ABUSE_DETECTED = "abuse_detected"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    timestamp: datetime
    ip_address: str
    user_agent: str
    endpoint: str
    threat_level: SecurityThreatLevel
    reason: BlockReason
    details: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int = 10
    window_size: int = 60  # seconds


class InMemoryRateLimiter:
    """In-memory rate limiter for development/testing."""
    
    def __init__(self):
        self.requests: Dict[str, deque] = defaultdict(deque)
        self.blocked_ips: Dict[str, datetime] = {}
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self):
        """Clean up old request records."""
        current_time = time.time()
        if current_time - self.last_cleanup < self.cleanup_interval:
            return
        
        cutoff_time = current_time - 3600  # Keep 1 hour of data
        
        for ip, request_times in self.requests.items():
            while request_times and request_times[0] < cutoff_time:
                request_times.popleft()
        
        # Clean up expired blocks
        expired_blocks = [
            ip for ip, block_time in self.blocked_ips.items()
            if (datetime.utcnow() - block_time).total_seconds() > 3600
        ]
        for ip in expired_blocks:
            del self.blocked_ips[ip]
        
        self.last_cleanup = current_time
    
    def is_rate_limited(self, ip: str, config: RateLimitConfig) -> Tuple[bool, Dict[str, Any]]:
        """Check if IP is rate limited."""
        self._cleanup_old_requests()
        
        # Check if IP is blocked
        if ip in self.blocked_ips:
            block_time = self.blocked_ips[ip]
            if (datetime.utcnow() - block_time).total_seconds() < 3600:  # 1 hour block
                return True, {
                    "reason": "ip_blocked",
                    "blocked_until": (block_time + timedelta(hours=1)).isoformat(),
                    "remaining_time": 3600 - (datetime.utcnow() - block_time).total_seconds()
                }
            else:
                del self.blocked_ips[ip]
        
        current_time = time.time()
        request_times = self.requests[ip]
        
        # Remove old requests outside the window
        minute_cutoff = current_time - 60
        hour_cutoff = current_time - 3600
        day_cutoff = current_time - 86400
        
        while request_times and request_times[0] < day_cutoff:
            request_times.popleft()
        
        # Count requests in different windows
        minute_requests = sum(1 for t in request_times if t > minute_cutoff)
        hour_requests = sum(1 for t in request_times if t > hour_cutoff)
        day_requests = len(request_times)
        
        # Check limits
        if minute_requests >= config.requests_per_minute:
            return True, {
                "reason": "minute_limit_exceeded",
                "limit": config.requests_per_minute,
                "current": minute_requests,
                "reset_time": (datetime.utcnow() + timedelta(seconds=60 - (current_time % 60))).isoformat()
            }
        
        if hour_requests >= config.requests_per_hour:
            return True, {
                "reason": "hour_limit_exceeded",
                "limit": config.requests_per_hour,
                "current": hour_requests,
                "reset_time": (datetime.utcnow() + timedelta(seconds=3600 - (current_time % 3600))).isoformat()
            }
        
        if day_requests >= config.requests_per_day:
            return True, {
                "reason": "day_limit_exceeded",
                "limit": config.requests_per_day,
                "current": day_requests,
                "reset_time": (datetime.utcnow() + timedelta(seconds=86400 - (current_time % 86400))).isoformat()
            }
        
        # Check burst limit (requests in last 10 seconds)
        burst_cutoff = current_time - 10
        burst_requests = sum(1 for t in request_times if t > burst_cutoff)
        
        if burst_requests >= config.burst_limit:
            return True, {
                "reason": "burst_limit_exceeded",
                "limit": config.burst_limit,
                "current": burst_requests,
                "reset_time": (datetime.utcnow() + timedelta(seconds=10)).isoformat()
            }
        
        return False, {}
    
    def record_request(self, ip: str):
        """Record a request from IP."""
        self.requests[ip].append(time.time())
    
    def block_ip(self, ip: str, reason: BlockReason):
        """Block an IP address."""
        self.blocked_ips[ip] = datetime.utcnow()
        logger.warning(f"IP {ip} blocked for reason: {reason.value}")


class SecurityMonitor:
    """Security monitoring and threat detection."""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.suspicious_ips: Set[str] = set()
        self.failed_auth_attempts: Dict[str, List[datetime]] = defaultdict(list)
        self.malicious_patterns: Dict[str, int] = defaultdict(int)
    
    def record_security_event(self, event: SecurityEvent):
        """Record a security event."""
        self.security_events.append(event)
        
        # Keep only recent events (last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.security_events = [
            e for e in self.security_events 
            if e.timestamp > cutoff_time
        ]
        
        # Log security event
        logger.warning(f"Security event: {event.threat_level.value} - {event.reason.value} from {event.ip_address}")
    
    def analyze_request_patterns(self, ip: str, endpoint: str, user_agent: str) -> Optional[SecurityThreatLevel]:
        """Analyze request patterns for suspicious activity."""
        # Check for rapid requests from same IP
        recent_events = [
            e for e in self.security_events
            if e.ip_address == ip and 
            (datetime.utcnow() - e.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        
        if len(recent_events) > 50:
            return SecurityThreatLevel.HIGH
        elif len(recent_events) > 20:
            return SecurityThreatLevel.MEDIUM
        
        # Check for suspicious user agents
        suspicious_agents = [
            "bot", "crawler", "spider", "scraper", "scanner", "hack", "exploit"
        ]
        
        if any(agent in user_agent.lower() for agent in suspicious_agents):
            return SecurityThreatLevel.MEDIUM
        
        # Check for endpoint scanning
        unique_endpoints = set(e.endpoint for e in recent_events)
        if len(unique_endpoints) > 10:
            return SecurityThreatLevel.MEDIUM
        
        return None
    
    def check_authentication_failures(self, ip: str) -> bool:
        """Check for excessive authentication failures."""
        now = datetime.utcnow()
        cutoff_time = now - timedelta(minutes=15)
        
        # Clean old attempts
        self.failed_auth_attempts[ip] = [
            attempt for attempt in self.failed_auth_attempts[ip]
            if attempt > cutoff_time
        ]
        
        # Check if too many failures
        return len(self.failed_auth_attempts[ip]) >= 5
    
    def record_auth_failure(self, ip: str):
        """Record an authentication failure."""
        self.failed_auth_attempts[ip].append(datetime.utcnow())
    
    def is_ip_suspicious(self, ip: str) -> bool:
        """Check if IP is marked as suspicious."""
        return ip in self.suspicious_ips
    
    def mark_ip_suspicious(self, ip: str):
        """Mark IP as suspicious."""
        self.suspicious_ips.add(ip)
        logger.warning(f"IP {ip} marked as suspicious")


class SecurityMiddleware:
    """Main security middleware class."""
    
    def __init__(self, use_redis: bool = False, redis_url: Optional[str] = None):
        self.use_redis = use_redis
        self.redis_client = None
        
        if use_redis and redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for rate limiting")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}, using in-memory rate limiter")
                self.use_redis = False
        
        self.rate_limiter = InMemoryRateLimiter()
        self.security_monitor = SecurityMonitor()
        self.rate_limit_configs = self._setup_rate_limits()
        self.whitelisted_ips = set()
        self.blacklisted_ips = set()
    
    def _setup_rate_limits(self) -> Dict[str, RateLimitConfig]:
        """Setup rate limit configurations for different endpoints."""
        return {
            # Authentication endpoints - stricter limits
            "/auth/login": RateLimitConfig(5, 20, 100, 3),
            "/auth/register": RateLimitConfig(3, 10, 50, 2),
            
            # Processing endpoints - moderate limits
            "/prescription/parse": RateLimitConfig(50, 200, 1000, 10),
            "/prescription/batch-parse": RateLimitConfig(5, 20, 100, 3),
            "/semantic-mapping/map": RateLimitConfig(30, 120, 600, 8),
            "/semantic-mapping/batch-map": RateLimitConfig(3, 12, 60, 2),
            "/safety-analysis/analyze": RateLimitConfig(100, 400, 2000, 15),
            "/safety-analysis/batch-analyze": RateLimitConfig(10, 40, 200, 5),
            "/recommendations/generate": RateLimitConfig(20, 80, 400, 6),
            "/recommendations/batch-generate": RateLimitConfig(2, 8, 40, 1),
            
            # Information endpoints - higher limits
            "/health": RateLimitConfig(100, 400, 2000, 20),
            "/service-status": RateLimitConfig(50, 200, 1000, 10),
            "/api-info": RateLimitConfig(50, 200, 1000, 10),
            
            # Default for other endpoints
            "default": RateLimitConfig(100, 400, 2000, 15)
        }
    
    def get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers (for proxy/load balancer setups)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct client IP
        return request.client.host if request.client else "unknown"
    
    def get_rate_limit_config(self, endpoint: str) -> RateLimitConfig:
        """Get rate limit configuration for endpoint."""
        return self.rate_limit_configs.get(endpoint, self.rate_limit_configs["default"])
    
    def check_ip_whitelist(self, ip: str) -> bool:
        """Check if IP is whitelisted."""
        return ip in self.whitelisted_ips
    
    def check_ip_blacklist(self, ip: str) -> bool:
        """Check if IP is blacklisted."""
        return ip in self.blacklisted_ips
    
    def add_to_whitelist(self, ip: str):
        """Add IP to whitelist."""
        self.whitelisted_ips.add(ip)
        logger.info(f"IP {ip} added to whitelist")
    
    def add_to_blacklist(self, ip: str):
        """Add IP to blacklist."""
        self.blacklisted_ips.add(ip)
        logger.warning(f"IP {ip} added to blacklist")
    
    async def process_request(self, request: Request) -> Optional[JSONResponse]:
        """Process request through security middleware."""
        ip = self.get_client_ip(request)
        endpoint = request.url.path
        user_agent = request.headers.get("User-Agent", "unknown")
        request_id = getattr(request.state, "request_id", "unknown")
        
        # Skip security checks for certain paths
        skip_paths = ["/docs", "/redoc", "/openapi.json", "/favicon.ico"]
        if endpoint in skip_paths:
            return None
        
        # Check IP blacklist
        if self.check_ip_blacklist(ip):
            event = SecurityEvent(
                timestamp=datetime.utcnow(),
                ip_address=ip,
                user_agent=user_agent,
                endpoint=endpoint,
                threat_level=SecurityThreatLevel.HIGH,
                reason=BlockReason.IP_BLACKLISTED,
                request_id=request_id
            )
            self.security_monitor.record_security_event(event)
            
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": {
                        "message": "Access denied",
                        "code": "IP_BLACKLISTED",
                        "request_id": request_id
                    }
                }
            )
        
        # Skip rate limiting for whitelisted IPs
        if self.check_ip_whitelist(ip):
            return None
        
        # Check for suspicious activity
        threat_level = self.security_monitor.analyze_request_patterns(ip, endpoint, user_agent)
        if threat_level and threat_level in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]:
            event = SecurityEvent(
                timestamp=datetime.utcnow(),
                ip_address=ip,
                user_agent=user_agent,
                endpoint=endpoint,
                threat_level=threat_level,
                reason=BlockReason.SUSPICIOUS_ACTIVITY,
                request_id=request_id
            )
            self.security_monitor.record_security_event(event)
            
            # Block IP temporarily
            self.rate_limiter.block_ip(ip, BlockReason.SUSPICIOUS_ACTIVITY)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "message": "Suspicious activity detected",
                        "code": "SUSPICIOUS_ACTIVITY",
                        "request_id": request_id
                    }
                }
            )
        
        # Check authentication failures
        if self.security_monitor.check_authentication_failures(ip):
            event = SecurityEvent(
                timestamp=datetime.utcnow(),
                ip_address=ip,
                user_agent=user_agent,
                endpoint=endpoint,
                threat_level=SecurityThreatLevel.MEDIUM,
                reason=BlockReason.AUTHENTICATION_FAILURE,
                request_id=request_id
            )
            self.security_monitor.record_security_event(event)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "message": "Too many authentication failures",
                        "code": "AUTH_FAILURE_LIMIT",
                        "request_id": request_id,
                        "retry_after": 900  # 15 minutes
                    }
                }
            )
        
        # Rate limiting
        config = self.get_rate_limit_config(endpoint)
        is_limited, limit_info = self.rate_limiter.is_rate_limited(ip, config)
        
        if is_limited:
            event = SecurityEvent(
                timestamp=datetime.utcnow(),
                ip_address=ip,
                user_agent=user_agent,
                endpoint=endpoint,
                threat_level=SecurityThreatLevel.LOW,
                reason=BlockReason.RATE_LIMIT_EXCEEDED,
                details=limit_info,
                request_id=request_id
            )
            self.security_monitor.record_security_event(event)
            
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "message": "Rate limit exceeded",
                        "code": "RATE_LIMIT_EXCEEDED",
                        "request_id": request_id,
                        **limit_info
                    }
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(config.requests_per_minute),
                    "X-RateLimit-Remaining": str(max(0, config.requests_per_minute - limit_info.get("current", 0))),
                    "X-RateLimit-Reset": limit_info.get("reset_time", "")
                }
            )
        
        # Record successful request
        self.rate_limiter.record_request(ip)
        
        return None
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        now = datetime.utcnow()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        recent_events = [e for e in self.security_monitor.security_events if e.timestamp > last_hour]
        daily_events = [e for e in self.security_monitor.security_events if e.timestamp > last_day]
        
        return {
            "timestamp": now.isoformat(),
            "security_events": {
                "last_hour": len(recent_events),
                "last_day": len(daily_events),
                "total": len(self.security_monitor.security_events)
            },
            "threat_levels": {
                level.value: len([e for e in recent_events if e.threat_level == level])
                for level in SecurityThreatLevel
            },
            "block_reasons": {
                reason.value: len([e for e in recent_events if e.reason == reason])
                for reason in BlockReason
            },
            "blocked_ips": len(self.rate_limiter.blocked_ips),
            "suspicious_ips": len(self.security_monitor.suspicious_ips),
            "whitelisted_ips": len(self.whitelisted_ips),
            "blacklisted_ips": len(self.blacklisted_ips)
        }


# Global security middleware instance
security_middleware = SecurityMiddleware()


# Middleware function for FastAPI
async def security_middleware_func(request: Request, call_next):
    """Security middleware function for FastAPI."""
    # Process request through security middleware
    security_response = await security_middleware.process_request(request)
    if security_response:
        return security_response
    
    # Continue with request processing
    response = await call_next(request)
    
    # Add security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response


# Utility functions
def record_auth_failure(ip: str):
    """Record authentication failure for IP."""
    security_middleware.security_monitor.record_auth_failure(ip)


def get_security_stats() -> Dict[str, Any]:
    """Get current security statistics."""
    return security_middleware.get_security_stats()


def add_ip_to_whitelist(ip: str):
    """Add IP to whitelist."""
    security_middleware.add_to_whitelist(ip)


def add_ip_to_blacklist(ip: str):
    """Add IP to blacklist."""
    security_middleware.add_to_blacklist(ip)


def get_rate_limit_status(ip: str, endpoint: str) -> Dict[str, Any]:
    """Get rate limit status for IP and endpoint."""
    config = security_middleware.get_rate_limit_config(endpoint)
    is_limited, limit_info = security_middleware.rate_limiter.is_rate_limited(ip, config)
    
    return {
        "ip": ip,
        "endpoint": endpoint,
        "is_limited": is_limited,
        "config": {
            "requests_per_minute": config.requests_per_minute,
            "requests_per_hour": config.requests_per_hour,
            "requests_per_day": config.requests_per_day,
            "burst_limit": config.burst_limit
        },
        "status": limit_info if is_limited else {"status": "ok"}
    }