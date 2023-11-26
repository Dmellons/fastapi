from db_users import db 
from decouple import config
from typing import List, Union
from fastapi import Depends, FastAPI, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from json import loads
from appwrite_config import databases, client
from wakeonlan import send_magic_packet
import uvicorn
import os
import json

SECRET_KEY = config("SECRET_KEY")
ALGORITHM = config("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(config("ACCESS_TOKEN_EXPIRE_MINUTES"))

##
# Response Classes
##
# Moved to api_types.py

    


##
# Internal auth classes
##
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str or None = None

class User(BaseModel):
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None

class UserInDB(User):
    hashed_password: str

class Computer(BaseModel):
    name: str
    mac_address: str or None = None
    ip: str or None = None

##
# Server Context
##
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth_2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# sql_obj = core.build_sql_object()

app = FastAPI(
    title="Melston API",
    description="Melston Api Documentation"
)

origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:8000",
    "http://192.168.0.*" 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

##
# Helper functoins
##

def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)
    
def authenticate_user(db, username: str, password: str):
    user = get_user(db, username)
    if not user: return False
    if not verify_password(password, user.hashed_password): return False
    return user

def create_access_token(data: dict, expires_delta: timedelta or None = None):
    to_encode = data.copy()
    if expires_delta: 
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_auth(token: str = Depends(oauth_2_scheme)):
    credential_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credential_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credential_exception
    user = get_user(db, username=token_data.username)
    if user is None:
        raise credential_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_auth)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


##
# API Endpoints / Routes
##
@app.post("/token/", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
            )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    error_message = f"Unexpected error occured: {exc}"
    return JSONResponse(status_code=500, content={"detail": error_message})

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@app.get("/test/{id}/")
async def test(id: int):
    return {"message": "Hello World", "id": id}

@app.post("/network/wol/{name}/")
# async def send_wol(ip:str, auth=Depends(get_auth)):
async def send_wol(name:str):
    query = databases.list_documents(
        database_id=config('APPWRITE_PROJECT_ID'),
        collection_id=config('APPWRITE_COMPUTER_DATABASE_ID')
    )

    computers:List[Computer] = query['documents']
    try:
        for computer in computers:
            if computer['name'] == name:
                mac = computer['mac_address'].replace(':','')
                name = computer['name']
                ip = computer['ip']
                break
        
        send_magic_packet(mac)
        return_obj={
                "message":f"Successfully sent WOL to {name.title()}", 
                }
        print(return_obj)
        return json.dumps(return_obj)
    except Exception as e:
        return_obj = json.dumps({
            "message": f"Unable to find computer with name: {name}"
        })
        print(return_obj)
        return return_obj

if __name__ == '__main__':
    uvicorn.run("main:app", reload=True, host="192.168.0.8", port=5000, log_level='info')

