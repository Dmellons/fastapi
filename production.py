#!/usr/bin/env python3
"""
Production deployment script for Melston API
Optimized for resource management and CPU efficiency
"""

import os
import sys
import subprocess
import signal
import time
from pathlib import Path

# Production configuration
PRODUCTION_CONFIG = {
    "workers": 1,
    "worker_class": "uvicorn.workers.UvicornWorker",
    "worker_connections": 100,
    "max_requests": 1000,
    "max_requests_jitter": 100,
    "timeout": 30,
    "keepalive": 5,
    "bind": "0.0.0.0:8000",
    "preload_app": True,
    "log_level": "warning",  # Reduced logging for better performance
    "access_logfile": None,  # Disable access logs
    "error_logfile": "/var/log/melston-api/error.log",
    "pid_file": "/tmp/melston-api.pid",
    "daemon": False,
    "user": "david",
    "group": "david"
}

def check_dependencies():
    """Check if all required dependencies are available"""
    required_packages = [
        "fastapi", "uvicorn", "gunicorn", "appwrite", 
        "wakeonlan", "python-jose", "passlib"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    return True

def setup_logging():
    """Setup logging directories"""
    log_dir = Path("/var/log/melston-api")
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the user can write to the log directory
        os.chmod(log_dir, 0o755)
        return True
    except PermissionError:
        print(f"Cannot create log directory {log_dir}. Running without file logging.")
        PRODUCTION_CONFIG["error_logfile"] = None
        return True
    except Exception as e:
        print(f"Error setting up logging: {e}")
        return False

def check_environment():
    """Check environment variables and configuration"""
    required_env_vars = [
        "SECRET_KEY",
        "APPWRITE_PROJECT_ID",
        "APPWRITE_COMPUTER_DATABASE_ID"
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return False
    
    return True

def create_systemd_service():
    """Create a systemd service file for the API"""
    service_content = f"""[Unit]
Description=Melston API
After=network.target

[Service]
Type=exec
User=david
Group=david
WorkingDirectory=/home/david/fastapi
Environment=PATH=/home/david/fastapi/venv/bin
ExecStart=/home/david/fastapi/venv/bin/python production.py --start
ExecReload=/bin/kill -HUP $MAINPID
KillMode=mixed
TimeoutStopSec=5
PrivateTmp=true
Restart=on-failure
RestartSec=10

# Resource limits
MemoryMax=512M
CPUQuota=50%
TasksMax=100

[Install]
WantedBy=multi-user.target
"""
    
    service_file = Path("/etc/systemd/system/melston-api.service")
    try:
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        # Reload systemd and enable the service
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "melston-api"], check=True)
        
        print(f"Systemd service created at {service_file}")
        print("You can now use: sudo systemctl start melston-api")
        return True
    except PermissionError:
        print("Cannot create systemd service (requires sudo)")
        return False
    except Exception as e:
        print(f"Error creating systemd service: {e}")
        return False

def start_production_server():
    """Start the production server with optimized settings"""
    print("Starting Melston API in production mode...")
    
    # Build gunicorn command
    cmd = [
        "gunicorn",
        "main:app",
        f"--workers={PRODUCTION_CONFIG['workers']}",
        f"--worker-class={PRODUCTION_CONFIG['worker_class']}",
        f"--worker-connections={PRODUCTION_CONFIG['worker_connections']}",
        f"--max-requests={PRODUCTION_CONFIG['max_requests']}",
        f"--max-requests-jitter={PRODUCTION_CONFIG['max_requests_jitter']}",
        f"--timeout={PRODUCTION_CONFIG['timeout']}",
        f"--keepalive={PRODUCTION_CONFIG['keepalive']}",
        f"--bind={PRODUCTION_CONFIG['bind']}",
        f"--log-level={PRODUCTION_CONFIG['log_level']}",
    ]
    
    if PRODUCTION_CONFIG["preload_app"]:
        cmd.append("--preload")
    
    if PRODUCTION_CONFIG["access_logfile"] is None:
        cmd.append("--access-logfile=-")  # stdout
    else:
        cmd.append(f"--access-logfile={PRODUCTION_CONFIG['access_logfile']}")
    
    if PRODUCTION_CONFIG["error_logfile"]:
        cmd.append(f"--error-logfile={PRODUCTION_CONFIG['error_logfile']}")
    
    if PRODUCTION_CONFIG["pid_file"]:
        cmd.append(f"--pid={PRODUCTION_CONFIG['pid_file']}")
    
    # Set resource limits using ulimit
    try:
        # Limit memory usage (500MB)
        subprocess.run(["ulimit", "-v", "512000"], shell=True)
        # Limit CPU time (no more than 50% CPU)
        subprocess.run(["ulimit", "-t", "3600"], shell=True)
    except:
        pass  # ulimit might not be available
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Start the server
        process = subprocess.Popen(cmd)
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}, shutting down gracefully...")
            process.terminate()
            process.wait(timeout=30)
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Wait for the process
        return_code = process.wait()
        print(f"Server exited with code {return_code}")
        return return_code
        
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        process.terminate()
        return 0
    except Exception as e:
        print(f"Error starting server: {e}")
        return 1

def stop_server():
    """Stop the running server"""
    pid_file = PRODUCTION_CONFIG["pid_file"]
    if not os.path.exists(pid_file):
        print("No PID file found. Server may not be running.")
        return
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        print(f"Stopping server with PID {pid}")
        os.kill(pid, signal.SIGTERM)
        
        # Wait for graceful shutdown
        time.sleep(5)
        
        # Check if still running
        try:
            os.kill(pid, 0)  # Check if process exists
            print("Process still running, sending SIGKILL")
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            print("Server stopped successfully")
        
        # Clean up PID file
        os.remove(pid_file)
        
    except Exception as e:
        print(f"Error stopping server: {e}")

def check_server_status():
    """Check if the server is running"""
    pid_file = PRODUCTION_CONFIG["pid_file"]
    if not os.path.exists(pid_file):
        print("Server is not running (no PID file)")
        return False
    
    try:
        with open(pid_file, 'r') as f:
            pid = int(f.read().strip())
        
        # Check if process exists
        os.kill(pid, 0)
        print(f"Server is running with PID {pid}")
        return True
        
    except ProcessLookupError:
        print("PID file exists but process is not running")
        os.remove(pid_file)
        return False
    except Exception as e:
        print(f"Error checking server status: {e}")
        return False

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python production.py [start|stop|restart|status|install-service]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "start":
        if not check_dependencies():
            sys.exit(1)
        if not setup_logging():
            sys.exit(1)
        if not check_environment():
            sys.exit(1)
        
        sys.exit(start_production_server())
    
    elif command == "stop":
        stop_server()
    
    elif command == "restart":
        stop_server()
        time.sleep(2)
        if not check_dependencies():
            sys.exit(1)
        if not setup_logging():
            sys.exit(1)
        if not check_environment():
            sys.exit(1)
        sys.exit(start_production_server())
    
    elif command == "status":
        check_server_status()
    
    elif command == "install-service":
        create_systemd_service()
    
    else:
        print(f"Unknown command: {command}")
        print("Usage: python production.py [start|stop|restart|status|install-service]")
        sys.exit(1)

if __name__ == "__main__":
    main()