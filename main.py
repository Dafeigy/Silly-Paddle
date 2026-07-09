from paddleocr import PaddleOCRVL
from dotenv import load_dotenv
import os
from pathlib import Path
import time


output_dir = Path("./output")
load_dotenv()

API_KEY = os.environ['API_KEY']

file_path = "./ModelRouter.pdf"
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="https://api.siliconflow.cn/v1",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
    vl_rec_api_key=API_KEY,
)

print("Start predicting...")
st = time.time()
output = pipeline.predict(file_path)
et = time.time()


final_res = "".join([res.markdown['markdown_texts'] for res in output])
print(final_res)

print(f"Predicting {file_path} done, time cost: {et - st} seconds")