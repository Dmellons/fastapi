from typing import List
from decouple import Config, RepositoryEnv
from icecream import ic
from datetime import datetime

from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.query import Query
from appwrite.id import ID

from plexapi.server import PlexServer


config = Config(RepositoryEnv('../../.env'))

# Plex Config
PLEX_TOKEN = config('PLEX_TOKEN')
PLEX_SERVER = config('PLEX_SERVER_BASE_URL')

plex = PlexServer(PLEX_SERVER, PLEX_TOKEN)

# Appwrite Config
APPWRITE_PROJECT_ID = config('NEXT_PUBLIC_APPWRITE_PROJECT_ID')
APPWRITE_PLEX_COLLECTION_ID = config('NEXT_PUBLIC_APPWRITE_PLEX_COLLECTION_ID')
APPWRITE_API_KEY = config('APPWRITE_API_KEY')
APPWRITE_URL_BASE = config('NEXT_PUBLIC_APPWRITE_ENDPOINT_URL')

appwrite_client = Client()
appwrite_client.set_endpoint(APPWRITE_URL_BASE)
appwrite_client.set_project(APPWRITE_PROJECT_ID)
appwrite_client.set_key(APPWRITE_API_KEY)

databases = Databases(appwrite_client)


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

def get_new_to_plex_library(sections: List[str] = ['Movies', 'TV Shows']) -> List[Media] | None:

    list_return_data = []
    for section in sections:
        existing_plex =  databases.list_documents(
            'watchlist',
            APPWRITE_PLEX_COLLECTION_ID, [
                Query.equal('content_type', content_section_translation[section]),
                Query.limit(600)
                
                
                ]
            )
        existing_section_data = [{'content_type': item['content_type'], 'tmdb_id': item['tmdb_id']} for item in existing_plex['documents'] if item['content_type'] == content_section_translation[section]]
        
        for media in plex.library.section(section).all():

            tmdb_id = [str(guid).split('tmdb://')[1][:-1] for guid in media.guids if 'tmdb' in str(guid)] if media.guids else None
            
            if tmdb_id is None or len(tmdb_id) == 0: continue

            tmdb_id = tmdb_id[0] if type(tmdb_id) == list else tmdb_id

            content_type = content_section_translation[section]
            
            if any(item for item in existing_section_data if ( item['tmdb_id'] == tmdb_id)):
                continue
            else:

                return_data = {
                        'title': media.title,
                        'tmdb_id': tmdb_id,
                        'content_type': content_type,
                        'date_added': media.addedAt.strftime("%Y-%m-%d")
                    }

                list_return_data.append(Media(**return_data))

    if len(list_return_data) > 0:
        ic(len(list_return_data))
        return list_return_data
               
    return None                


def update_plex_library( section:str = 'Movies') -> None:
    
    new_media = get_new_to_plex_library()
   
    if new_media is None: 
        ic('No new additions found')
        return 'No new additions found'
    for media in new_media:

        if type(media) != Media or media.tmdb_id is None : continue          
             
        tmdb_id = media.tmdb_id
 

        add_obj = {
            'title': media.title,
            'tmdb_id': tmdb_id,
            'content_type': media.content_type,

            'date_added': media.date_added
        }
  
        ic(add_obj)
        
        databases.create_document(
            'watchlist', 
            APPWRITE_PLEX_COLLECTION_ID, 
            ID.unique(), 
            add_obj,
            ['read("any")']
            )
        
        continue
    return f'Added {len(new_media)}'


if __name__ == '__main__':

    update_plex_library()

 