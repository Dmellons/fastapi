import signal
import sys
import time
from typing import List, Optional
from decouple import Config, RepositoryEnv
from icecream import ic
from datetime import datetime

from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from appwrite.id import ID

from plexapi.server import PlexServer
from plexapi.exceptions import PlexApiException, Unauthorized

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    print(f"\nShutdown signal {signum} received. Finishing current operation...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

config = Config(RepositoryEnv('../../.env'))

# Plex Config
PLEX_TOKEN = config('PLEX_TOKEN')
PLEX_SERVER = config('PLEX_SERVER_BASE_URL')

# Appwrite Config
APPWRITE_PROJECT_ID = config('NEXT_PUBLIC_APPWRITE_PROJECT_ID')
APPWRITE_PLEX_COLLECTION_ID = config('NEXT_PUBLIC_APPWRITE_PLEX_COLLECTION_ID')
APPWRITE_API_KEY = config('APPWRITE_API_KEY')
APPWRITE_URL_BASE = config('NEXT_PUBLIC_APPWRITE_ENDPOINT_URL')

# Performance settings
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
BATCH_SIZE = 50  # Process in smaller batches
REQUEST_DELAY = 0.1  # Small delay between requests to reduce CPU load

def create_plex_connection() -> Optional[PlexServer]:
    """Create Plex connection with error handling and retries"""
    for attempt in range(MAX_RETRIES):
        try:
            plex = PlexServer(PLEX_SERVER, PLEX_TOKEN, timeout=30)
            # Test connection
            _ = plex.library.sections()
            print(f"Plex connection established successfully")
            return plex
        except (PlexApiException, Unauthorized, Exception) as e:
            print(f"Plex connection attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print("Failed to establish Plex connection after all retries")
                return None

def create_appwrite_connection() -> Optional[Databases]:
    """Create Appwrite connection with error handling"""
    try:
        appwrite_client = Client()
        appwrite_client.set_endpoint(APPWRITE_URL_BASE)
        appwrite_client.set_project(APPWRITE_PROJECT_ID)
        appwrite_client.set_key(APPWRITE_API_KEY)
        
        databases = Databases(appwrite_client)
        
        # Test connection with a simple query
        test_query = databases.list_documents(
            'watchlist',
            APPWRITE_PLEX_COLLECTION_ID,
            [Query.limit(1)]
        )
        print("Appwrite connection established successfully")
        return databases
        
    except Exception as e:
        print(f"Failed to establish Appwrite connection: {e}")
        return None

# Initialize connections
plex = create_plex_connection()
databases = create_appwrite_connection()

if not plex or not databases:
    print("Failed to establish required connections. Exiting.")
    sys.exit(1)

content_section_translation = {
    'Movies': 'movie',
    'TV Shows': 'tv',
}
sections = [key for key in content_section_translation.keys()]

class Media:
    def __init__(self, tmdb_id: int, content_type: str, date_added: datetime, title: str) -> None:
        self.title = title
        self.tmdb_id = tmdb_id
        self.content_type = content_type
        self.date_added = date_added.strftime("%Y-%m-%d") if type(date_added) == datetime else date_added

def get_existing_media_efficiently(content_type: str) -> set:
    """Get existing media IDs more efficiently with pagination and limits"""
    existing_tmdb_ids = set()
    offset = 0
    batch_size = 100  # Reasonable batch size
    
    while not shutdown_requested:
        try:
            existing_plex = databases.list_documents(
                'watchlist',
                APPWRITE_PLEX_COLLECTION_ID, 
                [
                    Query.equal('content_type', content_type),
                    Query.limit(batch_size),
                    Query.offset(offset)
                ]
            )
            
            documents = existing_plex.get('documents', [])
            if not documents:
                break
                
            # Add TMDB IDs to our set
            for item in documents:
                if item.get('tmdb_id'):
                    existing_tmdb_ids.add(str(item['tmdb_id']))
            
            offset += batch_size
            
            # Small delay to reduce CPU load
            time.sleep(REQUEST_DELAY)
            
            # If we got fewer documents than requested, we've reached the end
            if len(documents) < batch_size:
                break
                
        except Exception as e:
            print(f"Error fetching existing media (offset {offset}): {e}")
            break
    
    print(f"Found {len(existing_tmdb_ids)} existing {content_type} entries")
    return existing_tmdb_ids

def get_new_to_plex_library(sections: List[str] = ['Movies', 'TV Shows']) -> List[Media] | None:
    """Get new media with improved performance and error handling"""
    if shutdown_requested:
        return None
        
    list_return_data = []
    
    for section in sections:
        if shutdown_requested:
            break
            
        print(f"Processing section: {section}")
        content_type = content_section_translation[section]
        
        # Get existing media efficiently
        existing_tmdb_ids = get_existing_media_efficiently(content_type)
        
        try:
            # Get Plex library section with error handling
            plex_section = plex.library.section(section)
            all_media = plex_section.all()
            print(f"Found {len(all_media)} items in Plex {section} section")
            
        except Exception as e:
            print(f"Error accessing Plex section {section}: {e}")
            continue
        
        processed_count = 0
        for media in all_media:
            if shutdown_requested:
                break
                
            try:
                # Extract TMDB ID safely
                tmdb_id = None
                if hasattr(media, 'guids') and media.guids:
                    for guid in media.guids:
                        guid_str = str(guid)
                        if 'tmdb' in guid_str and 'tmdb://' in guid_str:
                            try:
                                tmdb_id = guid_str.split('tmdb://')[1].rstrip('?')
                                break
                            except (IndexError, AttributeError):
                                continue
                
                if not tmdb_id:
                    continue
                
                # Check if already exists
                if tmdb_id in existing_tmdb_ids:
                    continue
                
                # Create new media entry
                return_data = {
                    'title': getattr(media, 'title', 'Unknown'),
                    'tmdb_id': tmdb_id,
                    'content_type': content_type,
                    'date_added': getattr(media, 'addedAt', datetime.now()).strftime("%Y-%m-%d") if hasattr(media, 'addedAt') else datetime.now().strftime("%Y-%m-%d")
                }
                
                list_return_data.append(Media(**return_data))
                processed_count += 1
                
                # Process in batches to avoid memory issues
                if processed_count % BATCH_SIZE == 0:
                    print(f"Processed {processed_count} items from {section}")
                    time.sleep(REQUEST_DELAY)  # Small delay to reduce CPU load
                
            except Exception as e:
                print(f"Error processing media item in {section}: {e}")
                continue
        
        print(f"Completed processing {section}: {processed_count} items processed")
    
    if len(list_return_data) > 0:
        ic(f"Found {len(list_return_data)} new items")
        return list_return_data
               
    return None

def update_plex_library_efficiently() -> str:
    """Update Plex library with improved error handling and performance"""
    if shutdown_requested:
        return "Shutdown requested"
        
    print("Starting Plex library update...")
    
    try:
        new_media = get_new_to_plex_library()
    except Exception as e:
        print(f"Error getting new media: {e}")
        return f"Error getting new media: {e}"
   
    if new_media is None: 
        ic('No new additions found')
        return 'No new additions found'
    
    added_count = 0
    failed_count = 0
    
    for i, media in enumerate(new_media):
        if shutdown_requested:
            print("Shutdown requested during update process")
            break
            
        if type(media) != Media or media.tmdb_id is None:
            continue          
             
        tmdb_id = media.tmdb_id
        
        add_obj = {
            'title': media.title,
            'tmdb_id': tmdb_id,
            'content_type': media.content_type,
            'date_added': media.date_added
        }
        
        try:
            databases.create_document(
                'watchlist', 
                APPWRITE_PLEX_COLLECTION_ID, 
                ID.unique(), 
                add_obj,
                ['read("any")']
            )
            
            added_count += 1
            ic(f"Added: {add_obj['title']} ({added_count}/{len(new_media)})")
            
            # Small delay between database writes to reduce load
            time.sleep(REQUEST_DELAY)
            
        except Exception as e:
            failed_count += 1
            print(f"Failed to add {media.title}: {e}")
            continue
    
    result_message = f'Added {added_count} items'
    if failed_count > 0:
        result_message += f', {failed_count} failed'
    
    print(f"Update complete: {result_message}")
    return result_message

def run_with_monitoring():
    """Run the update with resource monitoring and limits"""
    start_time = time.time()
    max_runtime = 3600  # 1 hour maximum runtime
    
    try:
        result = update_plex_library_efficiently()
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"Operation completed in {runtime:.2f} seconds")
        print(f"Result: {result}")
        
        if runtime > max_runtime:
            print("Warning: Operation took longer than expected")
            
        return result
        
    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
        return "Interrupted by user"
    except Exception as e:
        print(f"Unexpected error during operation: {e}")
        return f"Error: {e}"

if __name__ == '__main__':
    print("Starting watchlist update with optimizations...")
    print("Press Ctrl+C to stop gracefully")
    
    # Check if connections are still valid
    if not plex or not databases:
        print("Invalid connections detected. Attempting to reconnect...")
        plex = create_plex_connection()
        databases = create_appwrite_connection()
        
        if not plex or not databases:
            print("Failed to reconnect. Exiting.")
            sys.exit(1)
    
    try:
        result = run_with_monitoring()
        print(f"Final result: {result}")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)