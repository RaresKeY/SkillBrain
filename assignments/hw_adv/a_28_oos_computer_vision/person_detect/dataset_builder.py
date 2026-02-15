import os
import re
import json
import concurrent.futures
from curl_cffi import requests

# Configuration
CONFIG_FILE = "dataset_config.json"
LIMIT_PER_TERM = 4
MAIN_OUTPUT_DIR = "person_face_dataset"
MAX_WORKERS = 10  # Adjust based on your network/CPU

def harvest_links(search_term, limit=5):
    url = f"https://www.pexels.com/search/videos/{search_term}/"
    print(f"Searching for '{search_term}' on {url}...")
    
    try:
        response = requests.get(url, impersonate="chrome110")
        if response.status_code != 200:
            print(f"Failed to fetch page: {response.status_code}")
            return []
        
        html = response.text
        pattern = r'href="/video/([a-zA-Z0-9-]+-\d+/)"'
        matches = re.findall(pattern, html)
        
        links = []
        seen = set()
        for match in matches:
            if len(links) >= limit:
                break
            full_link = f"https://www.pexels.com/video/{match}"
            if full_link not in seen:
                links.append(full_link)
                seen.add(full_link)
            
        print(f"Found {len(links)} links for '{search_term}'.")
        return links

    except Exception as e:
        print(f"Error harvesting links for '{search_term}': {e}")
        return []

def download_video(url, output_dir):
    import urllib.parse
    
    # Extract video ID from URL (e.g., .../video/slug-12345/ -> 12345)
    video_id_match = re.search(r'-(\d+)/?$', url)
    video_id = video_id_match.group(1) if video_id_match else "unknown"

    # Check if file with this ID already exists
    for filename in os.listdir(output_dir):
        if f"[{video_id}]" in filename:
            print(f"Skipping {url} (already exists: {filename})")
            return True

    print(f"Processing {url}...")
    
    try:
        # 1. Fetch Page HTML
        response = requests.get(url, impersonate="chrome110")
        if response.status_code != 200:
            print(f"Failed to fetch page {url}: {response.status_code}")
            return False
            
        html = response.text
        
        # 2. Extract Direct Video URL
        # Pattern: file-url=https%3A%2F%2Fvideos.pexels.com%2Fvideo-files%2F...
        pattern = r'file-url=(https%3A%2F%2Fvideos\.pexels\.com%2Fvideo-files%2F[^&"]+)'
        match = re.search(pattern, html)
        
        if not match:
            print(f"Could not extract video URL from {url}")
            return False
            
        encoded_url = match.group(1)
        video_url = urllib.parse.unquote(encoded_url)
        
        # 3. Determine Filename
        # We try to get a title from the page or URL, fallback to ID
        title_match = re.search(r'<title>(.*?)</title>', html)
        if title_match:
            title = title_match.group(1).split(' Pexels')[0].strip()
            # Sanitize title
            title = re.sub(r'[\\/*?:"<>|]', "", title)[:50]
        else:
            title = f"video_{video_id}"
            
        filename = f"{title} [{video_id}].mp4"
        output_path = os.path.join(output_dir, filename)
        
        # 4. Download File
        print(f"Downloading to {output_path}...")
        vid_response = requests.get(video_url, impersonate="chrome110", stream=True)
        
        if vid_response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in vid_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Done: {filename}")
            return True
        else:
            print(f"Failed to download video file: {vid_response.status_code}")
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

    print(f"  > Found {len(unique_links)} unique videos. Starting parallel download...")

    # Phase 2: Download Videos concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_url = {executor.submit(download_video, url, target_dir): url for url in unique_links}
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
        return

    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Starting dataset collection from {CONFIG_FILE}...")
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