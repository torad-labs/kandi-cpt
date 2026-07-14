import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

START_URL = "https://keyboard.university/"
OUTPUT_DIR = "/tmp/conductor-live-test-13399077463813043070/keyboard-data/"
visited = set()
queue = [START_URL]

def is_internal(url):
    return urlparse(url).netloc == urlparse(START_URL).netloc

print(f"Starting scrape of {START_URL}...")

while queue:
    url = queue.pop(0)
    if url in visited:
        continue
    
    try:
        print(f"Fetching: {url}")
        res = requests.get(url, timeout=10)
        visited.add(url)
        if res.status_code != 200:
            continue
            
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Save page content as text
        path = urlparse(url).path
        if not path or path == "/":
            path = "/index.html"
        
        # Create local directory structure
        local_path = os.path.join(OUTPUT_DIR, path.lstrip('/'))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(soup.get_text(separator='\n'))
            
        # Find more links
        for link in soup.find_all('a', href=True):
            full_url = urljoin(url, link['href'])
            if is_internal(full_url) and full_url not in visited:
                queue.append(full_url)
                
    except Exception as e:
        print(f"Error scraping {url}: {e}")

print(f"Done. Data saved to {OUTPUT_DIR}")
