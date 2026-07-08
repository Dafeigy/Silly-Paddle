"""
PaddleOCR-VL RESTful 服务

支持三种输入方式：
1. POST /ocr/base64  — JSON body 传 base64 字符串
2. POST /ocr/upload  — multipart/form-data 直接上传文件
3. GET  /health      — 健康检查

base64 入参通过 fileType 区分文件类型：0 表示 PDF，1 表示图片。
"""

import base64
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from paddleocr import PaddleOCRVL

# ─── 配置 ───────────────────────────────────────────────────────
load_dotenv()
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError("请在 .env 文件中设置 API_KEY")

OUTPUT_DIR = Path("./output")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ocr-service")

# ─── Pipeline 生命周期 ───────────────────────────────────────────
pipeline: Optional[PaddleOCRVL] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    logger.info("正在初始化 PaddleOCR-VL pipeline ...")
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url="https://api.siliconflow.cn/v1",
        vl_rec_api_model_name="PaddlePaddle/PaddleOCR-VL-1.5",
        vl_rec_api_key=API_KEY,
    )
    logger.info("Pipeline 就绪")
    yield
    logger.info("服务关闭")


app = FastAPI(
    title="PaddleOCR-VL Service",
    version="1.0.0",
    lifespan=lifespan,
)

# ─── Request / Response 模型 ─────────────────────────────────────


class Base64Request(BaseModel):
    """base64 请求体"""

    payload: str = Field(
        ...,
        description="文件的 base64 编码字符串（支持 data URI 前缀 `data:<mime>;base64,`）",
        examples=["iVBORw0KGgo...", "data:image/png;base64,iVBORw0KGgo..."],
    )
    fileType: int = Field(
        ...,
        ge=0,
        le=1,
        description="文件类型：0 表示 PDF，1 表示图片",
        examples=[1],
    )


class OCRResult(BaseModel):
    """单页结果"""

    page_index: int
    markdown_text: str


class OCRResponse(BaseModel):
    """OCR 响应体"""

    success: bool
    page_count: int
    elapsed_seconds: float
    results: list[OCRResult]


class HealthResponse(BaseModel):
    status: str
    version: str


# ─── 工具函数 ────────────────────────────────────────────────────

# 常见图片类型的 magic bytes 签名
_MAGIC_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # 需要进一步检查 WEBP 标记
    b"BM": "image/bmp",
}

PDF_SIGNATURE = b"%PDF"


def detect_mime_type(raw: bytes) -> str:
    """通过文件头 magic bytes 自动检测 MIME 类型"""
    if raw[:4] == PDF_SIGNATURE:
        return "application/pdf"
    for signature, mime in _MAGIC_SIGNATURES.items():
        if raw[: len(signature)] == signature:
            return mime
    # 兜底：让 cv2.imdecode 尝试
    return "image/unknown"


def strip_data_uri(payload: str) -> tuple[str, Optional[str]]:
    """
    去除 data URI 前缀（如有），返回 (纯base64, 前缀中声明的mime或None)。

    data URI 格式: data:[<mediatype>][;base64],<data>
    """
    if not payload.startswith("data:"):
        return payload, None

    comma_idx = payload.find(",")
    if comma_idx == -1:
        raise ValueError("无效的 data URI：缺少逗号分隔符")

    header = payload[:comma_idx]  # data:image/png;base64
    pure_b64 = payload[comma_idx + 1 :]

    # 从 header 提取 mime type
    mime = None
    if header.startswith("data:"):
        mediatype = header[5:]  # image/png;base64
        if ";" in mediatype:
            mediatype = mediatype.split(";")[0]
        if mediatype:
            mime = mediatype

    return pure_b64, mime


def decode_b64_to_image(b64_payload: str) -> np.ndarray:
    """base64 → BGR numpy array（PNG / JPEG / WebP 等图片格式）"""
    b64_payload, _ = strip_data_uri(b64_payload)
    img_bytes = base64.b64decode(b64_payload)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("无法将 base64 数据解码为图片，请检查数据完整性")
    return img


def write_b64_to_temp_pdf(b64_payload: str) -> str:
    """base64 → 临时 PDF 文件路径，由 PaddleOCR pipeline 原生处理。"""
    b64_payload, _ = strip_data_uri(b64_payload)
    pdf_bytes = base64.b64decode(b64_payload)

    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(pdf_bytes)
        return tmp.name
    finally:
        tmp.close()


# ─── 核心预测逻辑 ────────────────────────────────────────────────


def do_predict(payload: str, file_type: int) -> list:
    """
    统一的预测入口：

    - 图片 (image/*) → base64 解码为 ndarray → pipeline.predict(ndarray)
    - PDF → base64 解码为临时文件 → pipeline.predict(path)
    """
    if file_type == 0:
        logger.info("输入类型: PDF")
        tmp_path = write_b64_to_temp_pdf(payload)
        try:
            return list(pipeline.predict(tmp_path))
        finally:
            os.unlink(tmp_path)

    if file_type == 1:
        logger.info("输入类型: image")
        img = decode_b64_to_image(payload)
        logger.info(f"图片尺寸: {img.shape[1]}x{img.shape[0]}")
        return list(pipeline.predict(img))

    raise ValueError("fileType 只支持 0 或 1：0 表示 PDF，1 表示图片")


# ─── 路由 ────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="1.0.0")


@app.post("/ocr/base64", response_model=OCRResponse)
async def ocr_base64(req: Base64Request):
    """
    通过 base64 字符串进行 OCR。

    支持的格式：PNG、JPEG、WebP、BMP、PDF（多页自动逐页识别）。
    JSON 中 fileType=0 表示 PDF，fileType=1 表示图片。
    支持纯 base64 字符串或 `data:<mime>;base64,<data>` 格式的 data URI。
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    try:
        st = time.perf_counter()
        results = do_predict(req.payload, req.fileType)
        elapsed = time.perf_counter() - st

        ocr_results = []
        for i, res in enumerate(results):
            md_text = "".join(res.markdown.get("markdown_texts", []))
            ocr_results.append(OCRResult(page_index=i, markdown_text=md_text))

        return OCRResponse(
            success=True,
            page_count=len(ocr_results),
            elapsed_seconds=round(elapsed, 3),
            results=ocr_results,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("OCR 处理失败")
        raise HTTPException(status_code=500, detail=f"OCR 处理异常: {e}")


@app.post("/ocr/upload", response_model=OCRResponse)
async def ocr_upload(file: UploadFile = File(...)):
    """
    直接上传文件进行 OCR。

    支持 PNG / JPEG / WebP / BMP / PDF 格式。
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline 尚未初始化")

    try:
        raw = await file.read()
        mime = detect_mime_type(raw)

        if mime == "application/pdf":
            # PDF：写入临时文件，用 pipeline 原生路径处理
            tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            try:
                tmp.write(raw)
                tmp_path = tmp.name
            finally:
                tmp.close()

            try:
                st = time.perf_counter()
                results = list(pipeline.predict(tmp_path))
                elapsed = time.perf_counter() - st
            finally:
                os.unlink(tmp_path)

        else:
            # 图片：numpy 解码 → 直接传 ndarray
            nparr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(
                    status_code=400, detail="无法解码上传的文件，请确认是有效的图片或 PDF"
                )

            st = time.perf_counter()
            results = list(pipeline.predict(img))
            elapsed = time.perf_counter() - st

        ocr_results = []
        for i, res in enumerate(results):
            md_text = "".join(res.markdown.get("markdown_texts", []))
            ocr_results.append(OCRResult(page_index=i, markdown_text=md_text))

        return OCRResponse(
            success=True,
            page_count=len(ocr_results),
            elapsed_seconds=round(elapsed, 3),
            results=ocr_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("文件上传 OCR 处理失败")
        raise HTTPException(status_code=500, detail=f"处理异常: {e}")


# ─── 启动入口 ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
