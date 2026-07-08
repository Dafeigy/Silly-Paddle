import base64
import requests
from dotenv import load_dotenv
import os

load_dotenv()
base_url = os.environ['BASE_URL']

import json

INPUT_FILE = "NEXT_STEP.pdf"
with open (INPUT_FILE, 'rb') as f:
    content = f.read()

base64content = b64_bytes = base64.b64encode(content).decode('ascii')
payload = {
    "payload": base64content,
    "fileType": 0
}
req = requests.post(f"{base_url}/ocr/base64", json=payload)

response = req.json()

with open ("pdf.json", "w") as f:
    f.write(json.dumps(response))

print(response['results'][0]['markdown_text'])
