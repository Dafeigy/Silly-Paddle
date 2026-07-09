import base64
import requests
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ['BASE_URL']

import json

INPUT_FILE = "Litellm.pdf"
with open (INPUT_FILE, 'rb') as f:
    content = f.read()

base64content = b64_bytes = base64.b64encode(content).decode('ascii')
payload = {
    "payload": base64content,
    "fileType": 0
}
req = requests.post(f"{base_url}/ocr/base64", json=payload)

response = req.json()

with open ("example1.json", "w", encoding='utf-8') as f:
    f.write(json.dumps(response, ensure_ascii=False, indent=4))

print(response['results'][0]['markdown_text'])
