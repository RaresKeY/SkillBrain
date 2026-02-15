import sys
import os
import re
import json
import concurrent.futures
from curl_cffi import requests

# Configuration
CONFIG_FILE = "dataset_config.json"
LIMIT_PER_TERM = 20
MAIN_OUTPUT_DIR = "person_face_dataset_images"
MAX_WORKERS = 10  # Adjust based on your network/CPU

def harvest_links(search_term, limit=5):
    # Changed to search for standard results (photos)
    url = f"https://www.pexels.com/search/{search_term}/"
    print(f"Searching for '{search_term}' on {url}...")
    
    try:
        response = requests.get(url, impersonate="chrome110")
        if response.status_code != 200:
            print(f"Failed to fetch page: {response.status_code}")
            return []
        
        html = response.text
        # Regex for photo links: href="/photo/slug-id/"
        pattern = r'href="/photo/([a-zA-Z0-9-]+-\d+/)"'
        matches = re.findall(pattern, html)
        
        links = []
        seen = set()
        for match in matches:
            if len(links) >= limit:
                break
            full_link = f"https://www.pexels.com/photo/{match}"
            if full_link not in seen:
                links.append(full_link)
                seen.add(full_link)
            
        print(f"Found {len(links)} links for '{search_term}'.")
        return links

    except Exception as e:
        print(f"Error harvesting links for '{search_term}': {e}")
        return []

def download_image(url, output_dir):
    import urllib.parse
    
    # Extract ID from URL (e.g., .../photo/slug-12345/ -> 12345)
    # Pexels photo slugs usually end with -ID
    id_match = re.search(r'-(\d+)/?$', url)
    image_id = id_match.group(1) if id_match else "unknown"

    # Check if file with this ID already exists
    for filename in os.listdir(output_dir):
        if f"[{image_id}]" in filename:
            print(f"Skipping {url} (already exists: {filename})")
            return True

    print(f"Processing {url}...")
    
    try:
        # 1. Fetch Page HTML to get Title and valid download context
        response = requests.get(url, impersonate="chrome110")
        if response.status_code != 200:
            print(f"Failed to fetch page {url}: {response.status_code}")
            return False
            
        html = response.text
        
        # 2. Determine Filename from Title
        title_match = re.search(r'<title>(.*?)</title>', html)
        if title_match:
            title = title_match.group(1).split(' Pexels')[0].strip()
            # Sanitize title
            title = re.sub(r'[\\/*?:"<<>>|]', "", title)[:50]
        else:
            title = f"photo_{image_id}"
            
        # 3. Construct Download URL
        # Attempt to use the direct download endpoint
        # Ensure url ends with / to append download cleanly or handle it
        clean_url = url.rstrip('/')
        download_url = f"{clean_url}/download/"
        
        # 4. Download File
        # We allow redirects because the download link usually redirects to the CDN
        img_response = requests.get(download_url, impersonate="chrome110", stream=True, allow_redirects=True)
        
        if img_response.status_code == 200:
            # Determine extension from headers or default to jpg
            content_type = img_response.headers.get('Content-Type', '')
            if 'png' in content_type:
                ext = 'png'
            elif 'webp' in content_type:
                ext = 'webp'
            else:
                ext = 'jpg'
                
            filename = f"{title} [{image_id}].{ext}"
            output_path = os.path.join(output_dir, filename)
            
            print(f"Downloading to {output_path}...")
            with open(output_path, 'wb') as f:
                for chunk in img_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Done: {filename}")
            return True
        else:
            print(f"Failed to download image file: {img_response.status_code}")
            # Fallback: Try finding the image source directly in HTML if download link fails
            # This is a basic fallback looking for the largest image
            # Regex for typical Pexels large image
            print("Trying fallback to direct src extraction...")
            src_pattern = r'src="(https://images\.pexels\.com/photos/[^"]+)"'
            src_matches = re.findall(src_pattern, html)
            if src_matches:
                # Usually the first one or one with large dimensions is good. 
                # We'll pick the first valid looking one and strip query params
                best_src = src_matches[0]
                # Strip params to get original/large
                if '?' in best_src:
                    best_src = best_src.split('?')[0]
                
                print(f"Fallback downloading from {best_src}...")
                fallback_response = requests.get(best_src, impersonate="chrome110", stream=True)
                if fallback_response.status_code == 200:
                    filename = f"{title} [{image_id}].jpg" # Assume jpg for fallback
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'wb') as f:
                        for chunk in fallback_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Done (Fallback): {filename}")
                    return True
            
            return False

    except Exception as e:
        print(f"Exception downloading {url}: {e}")
        return False

def process_keywords(keywords, base_folder, subfolder_name):
    target_dir = os.path.join(base_folder, subfolder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    unique_links = set()
    
    # Phase 1: Harvest Links concurrently
    print(f"  > Harvesting links for {len(keywords)} terms...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_term = {executor.submit(harvest_links, term, LIMIT_PER_TERM): term for term in keywords}
        for future in concurrent.futures.as_completed(future_to_term):
            try:
                links = future.result()
                unique_links.update(links)
            except Exception as e:
                term = future_to_term[future]
                print(f"Error harvesting term '{term}': {e}")

    print(f"  > Found {len(unique_links)} unique images. Starting parallel download...")

    # Phase 2: Download Images concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(download_image, url, target_dir): url for url in unique_links}
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                future.result()
            except Exception as e:
                print(f"Exception in download task: {e}")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    config_path = os.path.join(script_dir, CONFIG_FILE)
    
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Please create a 'dataset_config.json' file with your search terms.")
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Starting image dataset collection from {CONFIG_FILE}...")
    print(f"Main Output Directory: {MAIN_OUTPUT_DIR}")
    
    for dataset in config.get('datasets', []):
        folder_name = dataset.get('folder_name', 'unknown')
        category = dataset.get('category', 'General')
        test_case = dataset.get('test_case', 'Unknown')
        
        print(f"\nProcessing Dataset: {category} - {test_case}")
        print(f"Target Subfolder: {folder_name}")
        
        base_folder = os.path.join(script_dir, MAIN_OUTPUT_DIR, folder_name)
        
        # Process Positives
        if dataset.get('positive_keywords'):
            print(f"  > Collecting Positive Samples...")
            process_keywords(dataset['positive_keywords'], base_folder, "positive")
        
        # Process False Positives
        if dataset.get('false_positive_keywords'):
            print(f"  > Collecting False Positive Samples...")
            process_keywords(dataset['false_positive_keywords'], base_folder, "false_positive")
            
        # Process Negatives
        if dataset.get('negative_keywords'):
            print(f"  > Collecting Negative Samples...")
            process_keywords(dataset['negative_keywords'], base_folder, "negative")

    print("\nCollection complete.")

if __name__ == "__main__":
    main()
