import requests
from bs4 import BeautifulSoup
import json
import os

def scrape_plants():
    base_url = "https://example.com/plants"
    # This is a template scraper - in a real run, this would be 
    # tailored to the specific HTML structure of the site.
    print(f"Scraping {base_url}...")
    
    # Mocking the acquisition for the plate demo
    # In production, this would use requests.get() and BeautifulSoup
    data = [
        {"url": base_url + "/monstera", "text": "Monstera Deliciosa requires bright indirect light and watering every 1-2 weeks. Yellow leaves often indicate overwatering."},
        {"url": base_url + "/snake-plant", "text": "Snake Plant (Sansevieria) survives in low light and needs water only when soil is bone dry. Root rot is common if overwatered."},
    ]
    
    output_path = "data/raw/houseplants/corpus.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Successfully wrote {len(data)} entries to {output_path}")

if __name__ == "__main__":
    scrape_plants()
