import os
import pandas as pd
import requests
from urllib.parse import urlsplit
import re
from tqdm import tqdm

# Define function to sanitize file names
def sanitize_filename(filename):
    # Remove any invalid characters for filenames
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return filename

# Define function to download image
def download_image(url, folder="img"):
    if not url:  # Skip empty or None URLs
        print("Empty URL encountered. Skipping.")
        return
    
    try:
        # Send a GET request to the image URL
        response = requests.get(url, stream=True)
        
        # Check if the response status code is 200 (OK)
        if response.status_code == 200:
            # Extract the image filename from the URL
            image_name = os.path.basename(urlsplit(url).path)
            
            # If the image name is empty or None, generate a default filename
            if not image_name:
                image_name = "default_image.jpg"
                
            # Sanitize the filename to avoid issues with special characters
            image_name = sanitize_filename(image_name)
            
            # Create img folder if it doesn't exist
            if not os.path.exists(folder):
                os.makedirs(folder)

            # Define the path to save the image
            image_path = os.path.join(folder, image_name)

            # Print the full path to debug
            print(f"Saving image to: {image_path}")

            # Save the image content to the file system
            with open(image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            print(f"Image saved: {image_name}")
        else:
            print(f"Failed to download {url}: HTTP {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

# Read the CSV file
csv_file = 'C:/Users/alexi/Documents/archive_code/spirit_beauty/products.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_file)

# Ensure the 'Image src' column exists
if 'Image Src' not in df.columns:
    print("The 'Image src' column is not found in the CSV file.")
    print(df.columns)
else:
    # Loop through each URL in the 'Image src' column
    for url in tqdm(df['Image Src']):
        download_image(url)
