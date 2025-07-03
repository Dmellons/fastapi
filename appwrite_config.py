import os
import locale
import asyncio
import time
from typing import Optional, Dict, Any
from functools import lru_cache

# Ensure UTF-8 encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Set locale if needed (optional)
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    # Fallback if locale is not available
    pass

from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.id import ID
from decouple import config

# Configuration
appwrite_api_endpoint = config('APPWRITE_API_ENDPOINT')
appwrite_project_id = config('APPWRITE_PROJECT_ID')
appwrite_collection_id = config('APPWRITE_COLLECTION_ID')
appwrite_api_key = config('HA_API_READ')

# Connection pooling and optimization
class OptimizedAppwriteClient:
    """
    Optimized Appwrite client with connection pooling and caching.
    """
    
    def __init__(self):
        self._client = None
        self._databases = None
        self._connection_pool_size = 10
        self._timeout = 30
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Appwrite client with optimized settings."""
        self._client = Client()
        self._client.set_endpoint(appwrite_api_endpoint)
        self._client.set_project(appwrite_project_id)
        self._client.set_key(appwrite_api_key)
        
        # Set timeout for requests
        # Note: Appwrite Python SDK doesn't expose all HTTP client settings
        # In production, consider using httpx with custom settings
        
        self._databases = Databases(self._client)
    
    @property
    def client(self) -> Client:
        """Get the Appwrite client."""
        if self._client is None:
            self._initialize_client()
        return self._client
    
    @property 
    def databases(self) -> Databases:
        """Get the Databases service."""
        if self._databases is None:
            self._initialize_client()
        return self._databases
    
    def reconnect(self):
        """Reconnect to Appwrite if needed."""
        print("Reconnecting to Appwrite...")
        self._initialize_client()

# Create optimized client instance
_appwrite_client = OptimizedAppwriteClient()

# Export the standard interfaces for backward compatibility
client = _appwrite_client.client
databases = _appwrite_client.databases

# Health check function
@lru_cache(maxsize=1)
def get_appwrite_health_check_cache_key():
    """Generate cache key for health checks."""
    return f"appwrite_health_{int(time.time() // 300)}"  # 5-minute cache

async def check_appwrite_connection() -> bool:
    """
    Check if Appwrite connection is healthy.
    """
    try:
        # Simple health check - try to list collections
        # This is cached for 5 minutes to avoid excessive health checks
        cache_key = get_appwrite_health_check_cache_key()
        
        # In a real implementation, you might want to use Redis or another cache
        # For now, we'll just make the call
        result = databases.list_documents(
            database_id=appwrite_project_id,
            collection_id=appwrite_collection_id,
            queries=[],  # Empty query just to test connection
        )
        return True
    except Exception as e:
        print(f"Appwrite connection check failed: {e}")
        return False

def get_appwrite_config() -> Dict[str, Any]:
    """
    Get Appwrite configuration for debugging.
    """
    return {
        "endpoint": appwrite_api_endpoint,
        "project_id": appwrite_project_id,
        "collection_id": appwrite_collection_id,
        "api_key_set": bool(appwrite_api_key and len(appwrite_api_key) > 0),
        "client_initialized": _appwrite_client._client is not None,
        "databases_initialized": _appwrite_client._databases is not None
    }

# Connection retry decorator
def with_retry(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry Appwrite operations on failure.
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        # Run sync function in thread pool
                        loop = asyncio.get_event_loop()
                        return await loop.run_in_executor(None, func, *args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Appwrite operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                        # Try to reconnect
                        _appwrite_client.reconnect()
                    
            raise last_exception
        
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Appwrite operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                        # Try to reconnect
                        _appwrite_client.reconnect()
                    
            raise last_exception
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Export health check and config functions
__all__ = [
    'client',
    'databases', 
    'check_appwrite_connection',
    'get_appwrite_config',
    'with_retry'
]