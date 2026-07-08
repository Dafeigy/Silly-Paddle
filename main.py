from paddleocr import PaddleOCRVL
from dotenv import load_dotenv
import os
from pathlib import Path
import time


output_dir = Path("./output")
load_dotenv()

API_KEY = os.environ['API_KEY']
print(API_KEY)
pipeline = PaddleOCRVL(
    vl_rec_backend="vllm-server", 
    vl_rec_server_url="https://api.siliconflow.cn/v1",
    vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
    vl_rec_api_key=API_KEY,
)

print("Start predicting...")
st = time.time()
output = pipeline.predict("paddleocr_vl_demo.png")
et = time.time()
print("Predicting done, time cost: ", et - st, "seconds")

final_res = "".join([res.markdown['markdown_texts'] for res in output])
print(final_res)