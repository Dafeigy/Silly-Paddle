# PaddleOCR-VL RESTful Service

基于 [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR) + FastAPI 的文档 OCR 服务，通过 SiliconFlow API 调用 VLM 进行版面解析与文字识别。支持图片（PNG / JPEG / WebP / BMP）和 PDF 文件，base64 字符串或直接文件上传两种输入方式。

## Setup
```bash
# NVIDIA GPU（以 CUDA 12.6 为例）
python -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# x64 CPU
python -m pip install paddlepaddle==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# paddleocr
python -m pip install -U "paddleocr[doc-parser]" -i https://pypi.tuna.tsinghua.edu.cn/simple

```

## 架构设计

```
┌──────────────┐     ┌──────────────────────────────────────┐     ┌─────────────────────┐
│   Client     │────▶│  FastAPI (app.py)                     │────▶│  SiliconFlow API     │
│  base64/文件  │     │                                      │     │  PaddleOCR-VL-1.5   │
└──────────────┘     │  ┌─────────────────────────────────┐  │     └─────────────────────┘
                     │  │ 输入适配层                        │  │
                     │  │  ┌─ data URI 解析                 │  │
                     │  │  ├─ fileType 类型分发             │  │
                     │  │  └─ base64 → ndarray / 临时PDF路径 │  │
                     │  └──────────────┬──────────────────┘  │
                     │                 ▼                      │
                     │  ┌─────────────────────────────────┐  │
                     │  │ PaddleOCRVL pipeline             │  │
                     │  │  ┌─ DocPreprocessor (方向/矫正)   │  │
                     │  │  ├─ LayoutDetection (版面分析)    │  │
                     │  │  └─ VLRecognition (VLM识别)      │  │
                     │  └─────────────────────────────────┘  │
                     └──────────────────────────────────────┘
```

### 输入处理链路

```
字符串输入 (str)
  │
  ├─ 文件路径 (.png/.jpg/...) ────▶ pipeline.predict(path)
  │
  └─ 文件路径 (.pdf) ────────────▶ ImageBatchSampler → PDFReader → 逐页 ndarray → predict()
                                   (pipeline 内部自动处理)

ndarray 输入 (np.ndarray)
  │
  └─ BGR numpy 数组 ─────────────▶ ReadImage 透传 → predict()
                                   (零拷贝，不走磁盘)

base64 输入 (本服务新增)
  │
  ├─ fileType=1 ─▶ base64 解码 → cv2.imdecode → ndarray → predict()
  │
  └─ fileType=0 ─▶ base64 解码 → 临时 PDF 文件 → pipeline 原生 PDF 处理
```

### API 路由

| 方法 | 路由 | Content-Type | 说明 |
|------|------|-------------|------|
| `GET` | `/health` | — | 健康检查 + pipeline 状态 |
| `POST` | `/ocr/base64` | `application/json` | JSON body 传 base64 字符串 |
| `POST` | `/ocr/upload` | `multipart/form-data` | 直接上传原始文件 |

### 文件结构

```
siliconflow-paddle/
├── app.py                    # FastAPI 服务入口
├── main.py                   # 原始 cli 脚本（调试 / 对比用）
├── .env                      # 环境变量（API_KEY, 可选配置）
├── .gitignore
├── output/                   # 输出目录
└── README.md
```

## 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境 & 安装依赖
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS / Linux

pip install paddleocr fastapi uvicorn python-multipart python-dotenv opencv-python
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 启动服务

```bash
python app.py
```

服务默认监听 `http://0.0.0.0:4000`，`reload=True` 支持代码热更新。

成功启动后可以看到：

```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:ocr-service: 正在初始化 PaddleOCR-VL pipeline ...
INFO:ocr-service: Pipeline 就绪
INFO:     Application startup complete.
```

访问 http://localhost:4000/docs 可进入自动生成的 Swagger UI 交互文档。

### 4. 调用示例

#### 4.1 base64 图片

```bash
# 生成 base64
base64 -w0 scan.png > scan.b64

# 调用 API
curl -X POST http://localhost:4000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"payload": "'$(cat scan.b64)'", "fileType": 1}'
```

#### 4.2 带 data URI 前缀（自动推断格式）

```bash
curl -X POST http://localhost:4000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"payload": "data:image/png;base64,iVBORw0KGgo...", "fileType": 1}'
```

#### 4.3 base64 PDF

```bash
base64 -w0 report.pdf > report.b64

curl -X POST http://localhost:4000/ocr/base64 \
  -H "Content-Type: application/json" \
  -d '{"payload": "'$(cat report.b64)'", "fileType": 0}'
```

#### 4.4 上传文件

```bash
# 图片
curl -X POST http://localhost:4000/ocr/upload -F "file=@scan.png"

# PDF
curl -X POST http://localhost:4000/ocr/upload -F "file=@report.pdf"
```

#### 4.5 Python SDK 调用

```python
import base64
import requests

# 方式一：base64
with open("scan.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

resp = requests.post(
    "http://localhost:4000/ocr/base64",
    json={"payload": b64, "fileType": 1},
).json()

# 方式二：上传文件
with open("scan.png", "rb") as f:
    resp = requests.post(
        "http://localhost:4000/ocr/upload",
        files={"file": f},
    ).json()

# 拼接所有页的 Markdown 文本
full_md = "\n\n".join(page["markdown_text"] for page in resp["results"])
print(full_md)
```

### 5. 响应格式

```json
{
  "success": true,
  "page_count": 1,
  "elapsed_seconds": 3.214,
  "results": [
    {
      "page_index": 0,
      "markdown_text": "# 标题\n\n正文内容..."
    }
  ]
}
```

## 设计决策

| 决策 | 理由 |
|------|------|
| base64 端点在服务层解码，不侵入 pipeline | `predict()` 原生签名 `str \| ndarray \| list[ndarray]` 已覆盖所有场景，改第三方库升级时会被覆盖 |
| PDF 用临时文件路径中转 | 不在服务层引入 PDF 渲染库，直接复用 PaddleOCR pipeline 原生 PDF 处理能力；临时文件用完即删 |
| base64 payload 使用 `fileType` | `0` 表示 PDF，`1` 表示图片，避免靠 MIME 或 magic bytes 推断业务语义 |
| cv2.imdecode 解码图片 | 与 OpenCVImageReaderBackend 内部实现一致，BGR 格式对齐 ReadImage 的透传路径 |
