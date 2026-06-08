import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, quote

# --- CONFIGURATION ---
BASE_URL = "https://bulbapedia.bulbagarden.net/wiki/"
OUTPUT_DIR = "data/pokemon/00_raw"
# narrative-focused seeds
SEED_PAGES = [
    "Kanto", "Johto", "Hoenn", "Sinnoh", "Unova", "Kalos", "Alola", "Galar", "Paldea",
    "Ash_Ketchum", "Red_(character)", "Cynthia", "Giovanni", "Professor_Oak",
    "Team_Rocket", "Team_Galactic", "Team_Plasma", "Team_Flare", "Team_Skull", "Macro_Cosmos",
    "Pokémon_(anime)", "Pokémon_(manga)", "History_of_the_Pokémon_World"
]

# narrative-only filters
FORBIDDEN_PREFIXES = ("List_of_", "Category:")
FORBIDDEN_SUFFIXES = ("_(Pokémon)",)
FORBIDDEN_KEYWORDS = ("base_stats", "ability_list", "move_pool", "stat_table")

def is_narrative_page(title):
    if any(title.startswith(p) for p in FORBIDDEN_PREFIXES):
        return False
    if any(title.endswith(s) for s in FORBIDDEN_SUFFIXES):
        return False
    if any(k in title.lower() for k in FORBIDDEN_KEYWORDS):
        return False
    return True

def scrape_page(url):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Focus on the main content area of Bulbapedia
        content = soup.find(id="mw-content-text")
        if not content:
            return None
            
        # Remove script and style elements
        for script in content(["script", "style"]):
            script.decompose()
            
        return content.get_text(separator="\n", strip=True)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    visited = set()
    queue = list(SEED_PAGES)
    
    processed_count = 0
    max_pages = 500  # Safety limit for initial run
    
    print(f"Starting narrative scrape. Target: {OUTPUT_DIR}")
    
    while queue and processed_count < max_pages:
        page_title = queue.pop(0)
        if page_title in visited:
            continue
            
        url = urljoin(BASE_URL, page_title)
        text = scrape_page(url)
        
        if text:
            # Save to disk
            filename = f"{page_title.replace(' ', '_')}.txt"
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                f.write(text)
            
            visited.add(page_title)
            processed_count += 1
            print(f"[{processed_count}] Landed: {page_title}")
            
            # Find more narrative links to follow
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.startswith('/wiki/') and not href.startswith('/wiki/File:') and not href.startswith('/wiki/Category:'):
                    new_title = href.replace('/wiki/', '')
                    if is_narrative_page(new_title) and new_title not in visited:
                        queue.append(new_title)
        else:
            visited.add(page_title)

    print(f"Scrape complete. Landed {processed_count} narrative pages.")

if __name__ == "__main__":
    run()
