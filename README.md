# Melston API

A secure and scalable FastAPI-based API service that provides authentication, network management (Wake-on-LAN), and integration with various services including Appwrite, Home Assistant, and Plex.

## 🚀 Features

- **JWT Authentication**: Secure token-based authentication with refresh token support
- **Wake-on-LAN (WOL)**: Remote computer wake functionality with both authenticated and IP-based access
- **Multi-Service Integration**: 
  - Appwrite for database management
  - Home Assistant compatibility
  - Plex media server integration
- **Rate Limiting**: Built-in rate limiting for authentication endpoints
- **CORS Support**: Configurable cross-origin resource sharing
- **Comprehensive Error Handling**: Detailed error responses with proper HTTP status codes
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## 📋 Prerequisites

- Python 3.8+
- Appwrite instance (for database functionality)
- (Optional) Home Assistant instance
- (Optional) Plex Media Server

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd melston-api
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.template .env
   ```
   
   Edit `.env` with your configuration:
   ```env
   # Security
   SECRET_KEY="your-secret-key-here"
   ALGORITHM="HS256"
   ACCESS_TOKEN_EXPIRE_MINUTES=30
   
   # Home Assistant (Optional)
   HA_PW="home-assistant-password"
   HA_API_READ="home-assistant-api-key"
   
   # Appwrite Configuration
   APPWRITE_URL="http://your-appwrite-url"
   APPWRITE_API_ENDPOINT="http://your-appwrite-url/v1"
   APPWRITE_PROJECT_ID="your-project-id"
   APPWRITE_COMPUTER_DATABASE_ID="your-computer-db-id"
   APPWRITE_USER_DATABASE_ID="home_network"
   APPWRITE_COLLECTION_ID="computers"
   
   # CORS Configuration
   ALLOWED_ORIGINS="http://localhost:3000,http://localhost:8080"
   
   # Wake-on-LAN IP Whitelist
   ALLOWED_IPS="127.0.0.1,::1,192.168.1.100"
   
   # Plex Configuration (Optional)
   PLEX_TOKEN="your-plex-token"
   PLEX_SERVER_BASE_URL="http://your-plex-server:32400"
   ```

## 🚀 Running the Application

### Development Mode
```bash
python main.py
```

### Production Mode
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Or with Gunicorn:
```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Documentation

Once the server is running, access the interactive API documentation at:
- **Swagger UI**: `http://localhost:8000/api/v1/docs`
- **ReDoc**: `http://localhost:8000/api/v1/redoc`

## 🔑 Authentication

The API uses JWT (JSON Web Tokens) for authentication. 

### Getting an Access Token

**Option 1: OAuth2 Form Data**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/token" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "username=your-username&password=your-password"
```

**Option 2: JSON Request**
```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"username": "your-username", "password": "your-password"}'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### Using the Token
Include the token in the Authorization header:
```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     http://localhost:8000/api/v1/users/me
```

## 🖧 Wake-on-LAN Features

### Authenticated WOL
Send a Wake-on-LAN packet to a registered computer:
```bash
curl -X POST "http://localhost:8000/api/v1/network/wol" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"computer_name": "MyComputer"}'
```

### Home Assistant WOL (IP-based)
For Home Assistant integration (requires whitelisted IP):
```bash
curl -X POST "http://localhost:8000/network/wol/MyComputer"
```

### List Available Computers
```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     http://localhost:8000/api/v1/network/computers
```

## 🔧 Utility Scripts

### Generate Password Hash
Use the included utility to generate bcrypt password hashes:
```bash
python get_password_hash.py
```

### Update Plex Library (watchlist.py)
Synchronize your Plex library with Appwrite database:
```bash
python watchlist.py
```

## 📁 Project Structure

```
melston-api/
├── main.py                 # Main FastAPI application
├── appwrite_config.py      # Appwrite client configuration
├── db_users.py            # User database (example users)
├── get_password_hash.py   # Password hashing utility
├── watchlist.py           # Plex library synchronization
├── requirements.txt       # Python dependencies
├── .env.template          # Environment variables template
└── .gitignore            # Git ignore configuration
```

## 🔒 Security Features

- **Password Hashing**: Bcrypt with configurable rounds
- **JWT Tokens**: Separate access and refresh tokens
- **Rate Limiting**: Protects authentication endpoints
- **IP Whitelisting**: For Home Assistant WOL endpoints
- **CORS Protection**: Configurable allowed origins
- **Input Validation**: Pydantic models with validators

## 🧪 Testing

The API includes debug endpoints for troubleshooting:
```bash
curl http://localhost:8000/debug/computer/MyComputer
```

## 🐛 Troubleshooting

### Common Issues

1. **Unicode/Encoding Errors**
   - The application includes UTF-8 encoding fixes
   - Debug endpoints help identify encoding issues

2. **WOL Not Working**
   - Ensure MAC addresses are correctly formatted
   - Check if the computer supports Wake-on-LAN
   - Verify network configuration allows magic packets

3. **Authentication Failed**
   - Check if the user exists in `db_users.py`
   - Verify password hash is correct
   - Ensure JWT secret key is set

### Environment Variables
Missing environment variables will cause startup failures. Check:
- All required variables in `.env.template` are set
- No trailing spaces in variable values
- Proper formatting for lists (comma-separated)

## 📄 License

[Add your license information here]

## 👥 Contributing

[Add contributing guidelines here]

## 📞 Support

For issues and questions:
- Check the API documentation at `/api/v1/docs`
- Review debug endpoints for troubleshooting
- [Add contact/support information]

---

**Note**: This API is designed to work with specific infrastructure (Appwrite, Home Assistant, Plex). Ensure all required services are properly configured before deployment.