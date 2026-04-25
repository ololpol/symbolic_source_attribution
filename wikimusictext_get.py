import subprocess
import os
import json
import zipfile
import io

# Install the required packages if they are not installed
try:
    from unidecode import unidecode
except ImportError:
    subprocess.check_call(["python", '-m', 'pip', 'install', 'unidecode'])
    from unidecode import unidecode

try:
    from tqdm import tqdm
except ImportError:
    subprocess.check_call(["python", '-m', 'pip', 'install', 'tqdm'])
    from tqdm import tqdm

try:
    import requests
except ImportError:
    subprocess.check_call(["python", '-m', 'pip', 'install', 'requests'])
    import requests

def load_music(filename):
    # Convert the file to ABC notation
    p = subprocess.Popen(
        f'python {xml2abc_dir}/xml2abc.py -m 2 -c 6 -x "{filename}"',
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    out, err = p.communicate()

    output = out.decode('utf-8').replace('\r', '')  # Capture standard output
    music = unidecode(output).split('\n')

    return music

def download_and_extract(url):
    print(f"Downloading {url}")

    # Send an HTTP GET request to the URL and get the response
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        # Create a BytesIO object and write the HTTP response content into it
        zip_data = io.BytesIO()
        total_size = int(response.headers.get('content-length', 0))
        
        with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for data in response.iter_content(chunk_size=1024):
                pbar.update(len(data))
                zip_data.write(data)

        # Use the zipfile library to extract the file
        print("Extracting the zip file...")
        with zipfile.ZipFile(zip_data, "r") as zip_ref:
            zip_ref.extractall("")
        
        print("Done!")

    else:
        print("Failed to download the file. HTTP response code:", response.status_code)

# URL of the JSONL file
wikimt_url = "https://huggingface.co/datasets/sander-wood/wikimusictext/resolve/main/wikimusictext.jsonl"

# Local filename to save the downloaded file
local_filename = "wikimusictext.jsonl"

# Download the file and save it locally
response = requests.get(wikimt_url)
if response.status_code == 200:
    with open(local_filename, 'wb') as file:
        file.write(response.content)
    print(f"Downloaded '{local_filename}' successfully.")
else:
    print(f"Failed to download. Status code: {response.status_code}")

# Download the xml2abc.py script
# Visit https://wim.vree.org/svgParse/xml2abc.html
xml2abc_url = input("Enter the xml2abc URL: ")
xml2abc_url = "https://wim.vree.org/svgParse/xml2abc.py-174.zip"
download_and_extract(xml2abc_url)
xml2abc_dir = xml2abc_url.split('/')[-1][:-4].replace(".py", "").replace("-", "_")

# Download the Wikifonia dataset
# Visit http://www.synthzone.com/forum/ubbthreads.php/topics/384909/Download_for_Wikifonia_all_6,6
wikifonia_url = input("Enter the Wikifonia URL: ")
#wikifonia_url = http://www.synthzone.com/files/Wikifonia/Wikifonia.zip
download_and_extract(wikifonia_url)

# Correct the file extensions
for root, dirs, files in os.walk("Wikifonia"):
    for file in files:
        filepath = os.path.join(root, file)
        if filepath.endswith(".mxl"):
            continue
        else:
            new_filepath = filepath.split(".mxl")[0] + ".mxl"
            if os.path.exists(new_filepath):
                os.remove(new_filepath)
            os.rename(filepath, new_filepath)

wikimusictext = []
with open("wikimusictext.jsonl", "r", encoding="utf-8") as f:
    for line in f.readlines():
        wikimusictext.append(json.loads(line))

updated_wikimusictext = []

for song in tqdm(wikimusictext):
    filename = song["artist"] + " - " + song["title"] + ".mxl"
    filepath = os.path.join("Wikifonia", filename)
    song["music"] = load_music(filepath)
    updated_wikimusictext.append(song)

with open("wikimusictext.jsonl", "w", encoding="utf-8") as f:
    for song in updated_wikimusictext:
        f.write(json.dumps(song, ensure_ascii=False)+"\n")