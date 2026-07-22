from pathlib import Path
from typing import *

import requests
from tqdm import tqdm

__all__ = ["download_file", "download_bytes"]

  
def download_file(url: str, filepath: Union[str, Path], headers: dict = None, resume: bool = True) -> None:   
    # Ensure headers is a dict if not provided  
    headers = headers or {}  

    # Initialize local variables  
    file_path = Path(filepath)  
    downloaded_bytes = 0  

    # Check if we should resume the download  
    if resume and file_path.exists():  
        downloaded_bytes = file_path.stat().st_size  
        headers['Range'] = f"bytes={downloaded_bytes}-"  

    # Make a GET request to fetch the file  
    with requests.get(url, stream=True, headers=headers) as response:  
        response.raise_for_status()  # This will raise an HTTPError if the status is 4xx/5xx  

        # Calculate the total size to download  
        total_size = downloaded_bytes + int(response.headers.get('content-length', 0))  

        # Display a progress bar while downloading  
        with (
            tqdm(desc=f"Downloading {file_path.name}", total=total_size, unit='B', unit_scale=True, leave=False) as pbar,
            open(file_path, 'ab') as file, 
        ):  
            # Set the initial position of the progress bar  
            pbar.update(downloaded_bytes)  

            # Write the content to the file in chunks  
            for chunk in response.iter_content(chunk_size=4096):  
                file.write(chunk)  
                pbar.update(len(chunk))  
  

def download_bytes(url: str, headers: dict = None) -> bytes:  
    # Ensure headers is a dict if not provided  
    headers = headers or {}  

    # Make a GET request to fetch the file  
    with requests.get(url, stream=True, headers=headers) as response:  
        response.raise_for_status()  # This will raise an HTTPError if the status is 4xx/5xx  

        # Read the content of the response  
        return response.content  
  