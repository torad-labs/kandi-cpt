import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

START_URL = "https://keyboard.university/"
SAVE_DIR = "/tmp/conductor-live-test-13399077463813043070/keyboard-data/"

def scrape_wiki():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    visited = set()
    queue = [START_URL]
    
    while queue:
        url = queue.pop(0)
        if url in visited:
            continue
        
        print(f"Scraping: {url}")
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                continue
            
            visited.add(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (adjusting for common wiki structures)
            content = soup.find('main') or soup.find('article') or soup.find('body')
            if content:
                # Create a filename based on the URL path
                path = urlparse(url).path
                if not path or path == '/':
                    filename = "index.txt"
                else:
                    filename = path.strip('/').replace('/', '_') + ".txt"
                
                with open(os.path.join(SAVE_DIR, filename), 'w', encoding='utf-8') as f:
                    f.write(content.get_text(separator='\n', strip=True))
            
            # Find more internal links to follow
            for link in soup.find_all('a', href=True):
                full_url = urljoin(url, link['href'])
                if urlparse(full_url).netloc == urlparse(START_URL).netloc:
                    if full_url not in visited:
                        queue.append(full_url)
                        
        except Exception as e:
            print(f"Error scraping {url}: {e}")

if __name__ == "__main__":
    scrape_wiki()
