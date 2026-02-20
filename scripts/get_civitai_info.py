#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup
import re
import json
import sys
import io

# Set UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

# URLs to scrape
urls = [
    "https://civitai.com/models/929497?modelVersionId=2247497",
    "https://civitai.com/models/100435?modelVersionId=1096293",
    "https://civitai.com/models/1231943?modelVersionId=1736373"
]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

results = []

for url in urls:
    try:
        response = requests.get(url, headers=headers, timeout=30)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract full name from og:title
        name_tag = soup.find('meta', property='og:title')
        full_name = name_tag['content'].strip() if name_tag else "Unknown"
        
        # Extract UI name
        ui_name = re.sub(r'\s*v?\d+(\.\d+)*\s*', ' ', full_name)
        ui_name = re.sub(r'[^\w\s]', '', ui_name)
        ui_words = list(filter(None, ui_name.split()))
        ui_name = ' '.join(ui_words[:3])
        
        # Extract tags
        tags = []
        tag_elements = soup.find_all('a', href=re.compile(r'/tags/'))
        for elem in tag_elements:
            tag = elem.get_text(strip=True)
            if tag and len(tag) > 1 and tag not in tags:
                tags.append(tag)
        
        # Extract IDs from URL
        model_id = re.search(r'/models/(\d+)', url).group(1)
        version_id = re.search(r'modelVersionId=(\d+)', url).group(1)
        
        results.append({
            "model_id": model_id,
            "version_id": version_id,
            "full_name": full_name,
            "ui_name": ui_name,
            "tags": tags,
            "url": url
        })
        
    except Exception as e:
        print(f"Error with {url}: {e}")
        results.append({
            "model_id": "Unknown",
            "version_id": "Unknown",
            "full_name": "Error",
            "ui_name": "Error",
            "tags": [],
            "url": url
        })

# Save to file
with open('civitai_model_details.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Data saved to civitai_model_details.json")
