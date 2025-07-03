# main.py - Optimized FastAPI Server with Performance Improvements and Subnet Support
import os
import json
import time
import asyncio
import multiprocessing
import psutil
import gc
import threading
import ipaddress
from typing import List, Optional, Union, Annotated, Dict, Any
from datetime import datetime, timedelta, timezone
from contextlib import asynccontextmanager
from collections import defaultdict
from functools import wraps, lru_cache
from concurrent.futures import ThreadPoolExecutor

# Ensure UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status, Request, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, EmailStr, validator, field_validator
from jose import JWTError, jwt
from passlib.context import CryptContext
from wakeonlan import send_magic_packet
from decouple import config
from appwrite.query import Query

from db_users import db
from appwrite_config import databases, client

# Configuration
class Settings:
    SECRET_KEY: str = config("SECRET_KEY")
    ALGORITHM: str = config("ALGORITHM", default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(config("ACCESS_TOKEN_EXPIRE_MINUTES", default=30))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(config("REFRESH_TOKEN_EXPIRE_DAYS", default=7))
    API_VERSION: str = "v1"
    
    # Appwrite Configuration
    APPWRITE_URL: str = config('APPWRITE_URL', default='http://data.davidmellons.com')
    APPWRITE_API_ENDPOINT: str = config('APPWRITE_API_ENDPOINT', default='http://data.davidmellons.com/v1')
    APPWRITE_PROJECT_ID: str = config('APPWRITE_PROJECT_ID')
    APPWRITE_COMPUTER_DATABASE_ID: str = config('APPWRITE_COMPUTER_DATABASE_ID')
    APPWRITE_USER_DATABASE_ID: str = config('APPWRITE_USER_DATABASE_ID', default='home_network')
    APPWRITE_COLLECTION_ID: str = config('APPWRITE_COLLECTION_ID', default='computers')
    
    # Home Assistant Configuration (Optional)
    HA_PW: Optional[str] = config('HA_PW', default=None)
    HA_API_READ: Optional[str] = config('HA_API_READ', default=None)
    
    # Plex Configuration (Optional)
    PLEX_TOKEN: Optional[str] = config('PLEX_TOKEN', default=None)
    PLEX_SERVER_BASE_URL: Optional[str] = config('PLEX_SERVER_BASE_URL', default=None)
    
    # CORS settings - Load from environment variable
    ALLOWED_ORIGINS: List[str] = [
        origin.strip() for origin in config(
            'ALLOWED_ORIGINS', 
            default="http://localhost:3000,http://localhost:8000"
        ).split(',')
    ]
    
    # Allowed IPs for Wake-on-LAN - Load from environment variable
    ALLOWED_IPS: List[str] = [
        ip.strip() for ip in config(
            'ALLOWED_IPS', 
            default="127.0.0.1,::1"
        ).split(',')
    ]
    
    # Security settings
    BCRYPT_ROUNDS: int = 12
    MIN_PASSWORD_LENGTH: int = 8

settings = Settings()

# IP Subnet Support Function
def is_ip_allowed(client_ip: str, allowed_ips: List[str]) -> bool:
    """
    Check if client IP is allowed. Supports both exact IPs and CIDR notation.
    
    Examples:
    - 192.168.0.19 (exact IP)
    - 192.168.0.0/24 (subnet)
    - 10.0.0.0/8 (larger subnet)
    """
    try:
        client_addr = ipaddress.ip_address(client_ip)
        
        for allowed in allowed_ips:
            allowed = allowed.strip()
            
            # Check if it's a CIDR notation (contains '/')
            if '/' in allowed:
                try:
                    network = ipaddress.ip_network(allowed, strict=False)
                    if client_addr in network:
                        return True
                except (ipaddress.AddressValueError, ipaddress.NetmaskValueError):
                    # If CIDR parsing fails, treat as exact IP
                    if str(client_addr) == allowed:
                        return True
            else:
                # Exact IP match
                if str(client_addr) == allowed:
                    return True
                    
        return False
        
    except ipaddress.AddressValueError:
        # If client_ip is not a valid IP, fall back to exact string matching
        return client_ip in allowed_ips

# Production settings
class ProductionSettings:
    @staticmethod
    def get_worker_count():
        cpu_count = multiprocessing.cpu_count()
        return min(cpu_count, 8)  # Cap at 8 workers for memory usage
    
    WORKERS = get_worker_count()
    WORKER_CONNECTIONS = 1000
    BACKLOG = 2048
    KEEPALIVE = 2

# Thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/{settings.API_VERSION}/auth/token")
security = HTTPBearer(auto_error=False)

# Efficient Rate Limiter
class EfficientRateLimiter:
    def __init__(self):
        self.storage = defaultdict(list)
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # Clean up every 5 minutes
    
    def cleanup_old_entries(self):
        """Periodic cleanup instead of per-request cleanup"""
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            # Remove IPs with no recent requests
            to_remove = []
            for ip, timestamps in self.storage.items():
                # Keep only recent timestamps
                recent = [t for t in timestamps if now - t < 3600]  # Keep 1 hour
                if recent:
                    self.storage[ip] = recent
                else:
                    to_remove.append(ip)
            
            for ip in to_remove:
                del self.storage[ip]
            
            self.last_cleanup = now
    
    def is_allowed(self, client_ip: str, calls: int, period: int) -> bool:
        now = time.time()
        
        # Efficient cleanup only when needed
        if len(self.storage) > 1000:  # Only cleanup when storage gets large
            self.cleanup_old_entries()
        
        # Get recent requests for this IP
        timestamps = self.storage[client_ip]
        
        # Count recent requests (more efficient than list comprehension)
        recent_count = sum(1 for t in timestamps if now - t < period)
        
        if recent_count >= calls:
            return False
        
        # Add current request
        timestamps.append(now)
        
        # Keep only recent requests for this IP (limit per-IP storage)
        if len(timestamps) > calls * 2:  # Keep some buffer
            timestamps[:] = [t for t in timestamps if now - t < period]
        
        return True

# Global rate limiter instance
rate_limiter = EfficientRateLimiter()

# Database Manager with Caching
class DatabaseManager:
    def __init__(self):
        self.client = client
        self.databases = databases
        
        # Add simple caching for computer data
        self._computer_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
    async def get_computer_cached(self, computer_name: str):
        """Get computer data with caching."""
        now = time.time()
        cache_key = f"computer:{computer_name}"
        
        # Check cache first
        if cache_key in self._computer_cache:
            cached_data, cached_time = self._computer_cache[cache_key]
            if now - cached_time < self._cache_ttl:
                return cached_data
        
        # Query database in thread pool
        loop = asyncio.get_event_loop()
        
        try:
            query_result = await loop.run_in_executor(
                thread_pool,
                lambda: self.databases.list_documents(
                    database_id=settings.APPWRITE_PROJECT_ID,
                    collection_id=settings.APPWRITE_COMPUTER_DATABASE_ID,
                    queries=[Query.equal('name', computer_name)]
                )
            )
            
            # Cache the result
            self._computer_cache[cache_key] = (query_result, now)
            
            # Clean old cache entries periodically
            if len(self._computer_cache) > 100:
                self._clean_cache()
            
            return query_result
            
        except Exception as e:
            print(f"Database query error: {e}")
            raise
    
    async def list_computers_cached(self, skip: int = 0, limit: int = 100):
        """List computers with caching."""
        now = time.time()
        cache_key = f"computers_list:{skip}:{limit}"
        
        # Check cache first
        if cache_key in self._computer_cache:
            cached_data, cached_time = self._computer_cache[cache_key]
            if now - cached_time < self._cache_ttl:
                return cached_data
        
        # Query database in thread pool
        loop = asyncio.get_event_loop()
        
        try:
            query_result = await loop.run_in_executor(
                thread_pool,
                lambda: self.databases.list_documents(
                    database_id=settings.APPWRITE_PROJECT_ID,
                    collection_id=settings.APPWRITE_COMPUTER_DATABASE_ID,
                    queries=[
                        Query.offset(skip),
                        Query.limit(limit)
                    ]
                )
            )
            
            # Cache the result
            self._computer_cache[cache_key] = (query_result, now)
            
            return query_result
            
        except Exception as e:
            print(f"Database query error: {e}")
            raise
    
    def _clean_cache(self):
        """Remove old cache entries."""
        now = time.time()
        to_remove = [
            key for key, (_, cached_time) in self._computer_cache.items()
            if now - cached_time > self._cache_ttl
        ]
        for key in to_remove:
            del self._computer_cache[key]

# Create global database manager
db_manager = DatabaseManager()

# Pydantic models with validation
class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=100)

class UserCreate(UserBase):
    password: str = Field(..., min_length=settings.MIN_PASSWORD_LENGTH)
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not any(char.isdigit() for char in v):
            raise ValueError('Password must contain at least one digit')
        if not any(char.isupper() for char in v):
            raise ValueError('Password must contain at least one uppercase letter')
        return v

class User(UserBase):
    disabled: bool = False
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []

class Computer(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    mac_address: Optional[str] = Field(None, pattern="^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$")
    ip: Optional[str] = Field(None, pattern="^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")
    description: Optional[str] = None
    last_wake: Optional[datetime] = None

class WOLRequest(BaseModel):
    computer_name: str = Field(..., min_length=1, max_length=100)
    
class WOLResponse(BaseModel):
    success: bool
    message: str
    computer_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1)

# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up Melston API...")
    print(f"Available CPU cores: {multiprocessing.cpu_count()}")
    print(f"Thread pool workers: {thread_pool._max_workers}")
    yield
    # Shutdown
    print("Shutting down Melston API...")
    thread_pool.shutdown(wait=True)

# Create FastAPI app
app = FastAPI(
    title="Melston API",
    description="Melston API Documentation - A secure and scalable API service",
    version="1.0.0",
    lifespan=lifespan,
    docs_url=f"/api/{settings.API_VERSION}/docs",
    redoc_url=f"/api/{settings.API_VERSION}/redoc",
    openapi_url=f"/api/{settings.API_VERSION}/openapi.json"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        "*.davidmellons.com",
        "localhost",
        "127.0.0.1",
        "192.168.0.8",
        "data.davidmellons.com",
        "watchlist.davidmellons.com"]
)

# Request timing middleware
@app.middleware("http")
async def track_request_time(request: Request, call_next):
    """Track request processing time."""
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log slow requests
    if process_time > 1.0:  # Log requests taking more than 1 second
        print(f"Slow request: {request.method} {request.url.path} took {process_time:.2f}s")
    
    return response

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        },
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    # Log the error in production
    print(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "Internal server error",
                "status_code": 500,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        },
    )

# Async Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access"
    })
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "refresh"
    })
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

async def get_user_async(username: str) -> Optional[UserInDB]:
    """Async version of get_user."""
    await asyncio.sleep(0)  # Yield control
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    return None

async def authenticate_user_async(username: str, password: str) -> Union[UserInDB, bool]:
    """Async version of authenticate_user."""
    user = await get_user_async(username)
    if not user:
        return False
    
    # Run password verification in thread pool since it's CPU-intensive
    loop = asyncio.get_event_loop()
    is_valid = await loop.run_in_executor(
        thread_pool, 
        verify_password, 
        password, 
        user.hashed_password
    )
    
    if not is_valid:
        return False
    return user

async def create_tokens_async(username: str) -> dict:
    """Async token creation."""
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Run token creation in thread pool since JWT operations can be CPU-intensive
    loop = asyncio.get_event_loop()
    
    access_token = await loop.run_in_executor(
        thread_pool,
        create_access_token,
        {"sub": username},
        access_token_expires
    )
    
    refresh_token = await loop.run_in_executor(
        thread_pool,
        create_refresh_token,
        {"sub": username}
    )
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "access":
            raise credentials_exception
            
        token_data = TokenData(username=username)
        
    except JWTError:
        raise credentials_exception
        
    user = await get_user_async(username=token_data.username)
    if user is None:
        raise credentials_exception
        
    return User(**user.dict())

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """Ensure the current user is active."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Rate limiting decorator
def rate_limit(calls: int = 10, period: int = 60):
    """Efficient rate limiting decorator."""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            
            if not rate_limiter.is_allowed(client_ip, calls, period):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

# API Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Melston API",
        "version": "1.0.0",
        "docs": f"/api/{settings.API_VERSION}/docs"
    }

@app.get(f"/api/{settings.API_VERSION}/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Monitoring endpoints
@app.get(f"/api/{settings.API_VERSION}/monitoring/system", tags=["Monitoring"])
async def system_metrics():
    """Get system resource usage."""
    try:
        # Get CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)  # Non-blocking
        
        # Get memory usage
        memory = psutil.virtual_memory()
        
        # Get process-specific info
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return {
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "process": {
                "memory_rss": process_memory.rss,
                "memory_vms": process_memory.vms,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "connections": len(process.connections()) if hasattr(process, 'connections') else 0
            },
            "rate_limiter": {
                "active_ips": len(rate_limiter.storage),
                "total_requests": sum(len(timestamps) for timestamps in rate_limiter.storage.values())
            }
        }
    except Exception as e:
        return {"error": str(e)}

@app.get(f"/api/{settings.API_VERSION}/monitoring/performance", tags=["Monitoring"])
async def performance_metrics():
    """Get performance-related metrics."""
    try:
        # Garbage collection stats
        gc_stats = gc.get_stats()
        
        # Thread information
        thread_count = threading.active_count()
        
        return {
            "garbage_collection": gc_stats,
            "threads": {
                "active_count": thread_count,
                "main_thread_alive": threading.main_thread().is_alive(),
                "thread_pool_workers": thread_pool._max_workers
            },
            "asyncio": {
                "loop_running": True  # If we get here, the loop is running
            },
            "database_cache": {
                "cached_items": len(db_manager._computer_cache)
            }
        }
    except Exception as e:
        return {"error": str(e)}

# Authentication endpoints
@app.post(f"/api/{settings.API_VERSION}/auth/token", response_model=Token, tags=["Authentication"])
@rate_limit(calls=5, period=60)
async def login(request: Request, form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    user = await authenticate_user_async(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    tokens = await create_tokens_async(user.username)
    return Token(**tokens)

@app.post(f"/api/{settings.API_VERSION}/auth/login", response_model=Token, tags=["Authentication"])
@rate_limit(calls=5, period=60)
async def login_json(request: Request, login_data: LoginRequest):
    """
    JSON-based login endpoint, get an access token for future requests.
    """
    user = await authenticate_user_async(login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    tokens = await create_tokens_async(user.username)
    return Token(**tokens)

@app.post(f"/api/{settings.API_VERSION}/auth/refresh", response_model=Token, tags=["Authentication"])
async def refresh_token(token: Annotated[str, Depends(oauth2_scheme)]):
    """Refresh access token using refresh token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "refresh":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    # Verify user still exists and is active
    user = await get_user_async(username)
    if not user or user.disabled:
        raise credentials_exception
    
    # Create new access token
    tokens = await create_tokens_async(username)
    return Token(**tokens)

# User endpoints
@app.get(f"/api/{settings.API_VERSION}/users/me", response_model=User, tags=["Users"])
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Get current user information."""
    return current_user

# Home Assistant Wake-on-LAN endpoint (IP-based authentication) - OPTIMIZED WITH SUBNET SUPPORT
@app.post("/network/wol/{computer_name}", response_model=WOLResponse, tags=["Network"])
async def send_wol_ha(computer_name: str, request: Request):
    """
    Optimized WOL endpoint for Home Assistant with subnet support.
    Supports both exact IPs and CIDR notation (e.g., 192.168.0.0/24).
    """
    
    client_ip = request.client.host
    
    # Use the new subnet-aware IP checking
    if not is_ip_allowed(client_ip, settings.ALLOWED_IPS):
        print(f"Access denied for IP: {client_ip} trying to wake {computer_name}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied for IP: {client_ip}"
        )
    
    try:
        # Ensure computer_name is clean ASCII
        computer_name = str(computer_name).strip()
        
        # Use cached database query
        query_result = await db_manager.get_computer_cached(computer_name)
        
        if not query_result['documents']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Computer '{computer_name}' not found"
            )
        
        computer = query_result['documents'][0]
        mac_address = computer.get('mac_address')
        
        if not mac_address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Computer '{computer_name}' has no MAC address configured"
            )
        
        # Validate and clean MAC address - ensure it's ASCII
        mac_address = str(mac_address).strip()
        clean_mac = mac_address.replace(':', '').replace('-', '').replace(' ', '').upper()
        
        if len(clean_mac) != 12 or not all(c in '0123456789ABCDEF' for c in clean_mac):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid MAC address format for computer '{computer_name}'"
            )
        
        # Send WOL packet in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, send_magic_packet, clean_mac)
        
        # Log the wake action with safe encoding
        try:
            print(f"Client ({client_ip}) sent WOL to {computer_name} (MAC: {clean_mac})")
        except UnicodeEncodeError:
            # Fallback for any encoding issues in logging
            safe_computer_name = computer_name.encode('ascii', 'replace').decode('ascii')
            print(f"Client ({client_ip}) sent WOL to {safe_computer_name} (MAC: {clean_mac})")
        
        # Return response with clean ASCII strings
        return WOLResponse(
            success=True,
            message=f"Wake-on-LAN packet sent successfully to {computer_name}",
            computer_name=computer_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Better error handling with safe encoding
        try:
            error_msg = str(e)
            safe_computer_name = computer_name
        except UnicodeEncodeError:
            error_msg = repr(e)  # Use repr to avoid encoding issues
            safe_computer_name = computer_name.encode('ascii', 'replace').decode('ascii')
        
        print(f"Error sending WOL to {safe_computer_name}: {error_msg}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send Wake-on-LAN packet"
        )

# Authenticated Network endpoints
@app.post(
    f"/api/{settings.API_VERSION}/network/wol", 
    response_model=WOLResponse,
    tags=["Network"],
    summary="Send Wake-on-LAN packet"
)
async def send_wol(
    wol_request: WOLRequest,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """
    Send Wake-on-LAN magic packet to a computer by name.
    """
    try:
        # Use cached database query
        query_result = await db_manager.get_computer_cached(wol_request.computer_name)
        
        if not query_result['documents']:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Computer '{wol_request.computer_name}' not found"
            )
        
        computer = query_result['documents'][0]
        mac_address = computer.get('mac_address')
        
        if not mac_address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Computer '{wol_request.computer_name}' has no MAC address configured"
            )
        
        # Validate and clean MAC address
        clean_mac = mac_address.replace(':', '').replace('-', '').replace(' ', '').upper()
        
        if len(clean_mac) != 12 or not all(c in '0123456789ABCDEF' for c in clean_mac):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid MAC address format for computer '{wol_request.computer_name}'"
            )
        
        # Send WOL packet in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(thread_pool, send_magic_packet, clean_mac)
        
        # Log the wake action
        print(f"User {current_user.username} sent WOL to {wol_request.computer_name}")
        
        return WOLResponse(
            success=True,
            message=f"Wake-on-LAN packet sent successfully",
            computer_name=wol_request.computer_name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error sending WOL to {wol_request.computer_name}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send Wake-on-LAN packet"
        )

@app.get(
    f"/api/{settings.API_VERSION}/network/computers",
    response_model=List[Computer],
    tags=["Network"],
    summary="List available computers"
)
async def list_computers(
    current_user: Annotated[User, Depends(get_current_active_user)],
    skip: int = 0,
    limit: int = 100
):
    """
    List all available computers for Wake-on-LAN.
    """
    try:
        query_result = await db_manager.list_computers_cached(skip=skip, limit=limit)
        
        computers = []
        for doc in query_result['documents']:
            computers.append(Computer(
                name=doc.get('name'),
                mac_address=doc.get('mac_address'),
                ip=doc.get('ip'),
                description=doc.get('description'),
                last_wake=doc.get('last_wake')
            ))
        
        return computers
        
    except Exception as e:
        print(f"Error listing computers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve computer list"
        )

# Debug endpoint for troubleshooting encoding issues
@app.get("/debug/computer/{computer_name}", tags=["Debug"])
async def debug_computer_data(computer_name: str):
    """Debug endpoint to inspect computer data encoding"""
    try:
        query_result = await db_manager.get_computer_cached(computer_name)
        
        if not query_result['documents']:
            return {"error": "Computer not found"}
        
        computer = query_result['documents'][0]
        
        # Debug each field
        debug_info = {}
        for key, value in computer.items():
            try:
                # Check if value can be encoded as ASCII
                str(value).encode('ascii')
                debug_info[key] = {
                    "value": value,
                    "type": type(value).__name__,
                    "ascii_safe": True
                }
            except UnicodeEncodeError as e:
                debug_info[key] = {
                    "value": repr(value),  # Use repr to show the actual characters
                    "type": type(value).__name__,
                    "ascii_safe": False,
                    "error": str(e)
                }
        
        return {
            "computer_name": computer_name,
            "debug_info": debug_info,
            "raw_document": computer
        }
        
    except Exception as e:
        return {
            "error": f"Debug failed: {repr(e)}",
            "computer_name": computer_name
        }

# Test endpoints (only in development)
if config("ENV", default="production") == "development":
    @app.get(f"/api/{settings.API_VERSION}/test", tags=["Test"])
    async def test_endpoint():
        """Test endpoint - only available in development."""
        return {"message": "Test successful", "environment": "development"}

if __name__ == "__main__":
    # Production configuration
    workers = ProductionSettings.get_worker_count()
    
    # Use gunicorn for production (recommended)
    if config("ENV", default="production") == "production":
        print(f"Starting FastAPI with {workers} workers")
        print("For production, consider using gunicorn:")
        print(f"gunicorn main:app -w {workers} -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:5000 --worker-connections 1000 --backlog 2048")
        
        uvicorn.run(
            "main:app",
            host=config("HOST", default="0.0.0.0"),
            port=int(config("PORT", default=5000)),
            workers=1,  # Use 1 for uvicorn, use gunicorn for multiple workers
            log_level=config("LOG_LEVEL", default="info"),
            worker_connections=ProductionSettings.WORKER_CONNECTIONS,
            backlog=ProductionSettings.BACKLOG,
            reload=False
        )
    else:
        # Development mode
        uvicorn.run(
            "main:app",
            host=config("HOST", default="127.0.0.1"),
            port=int(config("PORT", default=5000)),
            log_level="debug",
            reload=True
        )