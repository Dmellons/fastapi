import os
import locale

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

appwrite_api_endpoint = config('APPWRITE_API_ENDPOINT')
appwrite_project_id = config('APPWRITE_PROJECT_ID')
appwrite_collection_id = config('APPWRITE_COLLECTION_ID')
appwrite_api_key = config('HA_API_READ')

client = Client()
client.set_endpoint(appwrite_api_endpoint)
client.set_project(appwrite_project_id)
client.set_key(appwrite_api_key)
databases = Databases(client)