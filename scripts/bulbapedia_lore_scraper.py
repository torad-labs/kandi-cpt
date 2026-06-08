import os
import requests
import re
from pathlib import Path

# Configuration
BASE_URL = 'https://bulbapedia.bulbagarden.net/w/api.php'
OUTPUT_DIR = Path('data/pokemon-lore/00_raw')
CATEGORIES = [
    'Category:Characters',
    'Category:Regions',
    'Category:Organizations',
    'Category:Plot',
    'Category:Pokémon_lore',
]

def get_category_members(category):
    """Fetch all pages in a given Bulbapedia category."""
    pages = []
    params = {
        'action': 'query',
        'list': 'categorymembers',
        'cmtitle': category,
        'cmlimit': 'max',
        'format': 'json'
    }
    
    while True:
        response = requests.get(BASE_URL, params=params).json()
        members = response.get('query', {}).get('categorymembers', [])
        pages.extend([m['title'] for m in members])
        
        if 'continue' in response:
            params['cmcontinue'] = response['continue']['cmcontinue']
        else:
            break
            
    return pages

def is_lore_page(title):
    """
    Filter for lore pages. 
    Exclude: List of... or ...(Pokémon) stats/species pages.
    """
    # Exclude List pages
    if title.startswith('List:') or 'List of' in title:
        return False
    # Exclude Pokemon species pages (e.g., 'Pikachu (Pokémon)')
    if re.search(r'\(Pokémon\)$', title):
        return False
    # Exclude common non-lore meta pages
    if title.startswith('Category:') or title.startswith('File:') or title.startswith('Template:'):
        return False
        
    return True

def fetch_page_content(title):
    """Fetch the plain text content of a page."""
    params = {
        'action': 'query',
        'prop': 'extracts',
        'exextractformat': 'plain',
        'explaintext': True,
        'titles': title,
        'format': 'json'
    }
    response = requests.get(BASE_URL, params=params).json()
    pages = response.get('query', {}).get('pages', {})
    
    for page_id in pages:
        return pages[page_id].get('extract', '')
    return ''

def main():
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_lore_pages = set()
    print(f'Gathering pages from categories: {CATEGORIES}...')
    
    for cat in CATEGORIES:
        members = get_category_members(cat)
        print(f'Found {len(members)} members in {cat}')
        for member in members:
            if is_lore_page(member):
                all_lore_pages.add(member)
    
    print(f'Filtered down to {len(all_lore_pages)} lore-specific pages.')
    
    count = 0
    for title in all_lore_pages:
        print(f'Scraping {title}...', end='\r')
        content = fetch_page_content(title)
        if content:
            # Sanitize filename
            safe_title = re.sub(r'[\\/*?\"<>|]', '_', title)
            with open(OUTPUT_DIR / f'{safe_title}.txt', 'w', encoding='utf-8') as f:
                f.write(f'Title: {title}\n\n{content}')
            count += 1
            
    print(f'\nSuccessfully saved {count} lore documents to {OUTPUT_DIR}')

if __name__ == '__main__':
    main()
