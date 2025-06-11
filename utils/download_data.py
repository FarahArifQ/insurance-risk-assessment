import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """
    Download a file with progress bar
    """
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def setup_data():
    """
    Download required data files from an alternative source
    """
    base_url = "https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3960/988373/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1687654321&Signature=XYZ"
    
    files = {
        "train.csv": "data/train.csv",
        "test.csv": "data/test.csv"
    }
    
    for file_name, file_path in files.items():
        if not os.path.exists(file_path):
            download_file(f"{base_url}/{file_name}", file_path)
        else:
            print(f"{file_path} already exists, skipping download...")

if __name__ == "__main__":
    setup_data()
